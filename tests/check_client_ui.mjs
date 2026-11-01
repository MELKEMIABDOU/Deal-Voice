// Optional local-browser smoke test. Only accepts the model-free preview server.
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

const base = process.argv[2] || "http://127.0.0.1:8765";
assert(["127.0.0.1", "localhost"].includes(new URL(base).hostname));
const status = await (await fetch(`${base}/api/v1/status`)).json();
assert.equal(status.model, "Apercu interface - sans modele", "Never run this smoke test against real models/data.");

const output = path.resolve("reports/ui");
mkdirSync(output, { recursive: true });
const browser = spawn(process.env.DEAL_VOICE_TEST_CHROME || "C:/Program Files/Google/Chrome/Application/chrome.exe", [
  "--headless=new", "--disable-gpu", "--disable-background-networking", "--no-first-run",
  "--remote-debugging-port=0", `--user-data-dir=${mkdtempSync(path.join(tmpdir(), "dealvoice-ui-"))}`,
], { windowsHide: true, stdio: ["ignore", "ignore", "pipe"] });

const endpoint = await new Promise((resolve, reject) => {
  const timer = setTimeout(() => reject(new Error("Browser did not start.")), 15000);
  browser.once("error", reject);
  browser.stderr.on("data", (chunk) => {
    const found = String(chunk).match(/DevTools listening on (ws:\/\/\S+)/);
    if (found) { clearTimeout(timer); resolve(found[1]); }
  });
});
const socket = new WebSocket(endpoint);
await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
let sequence = 0;
const pending = new Map();
socket.onmessage = ({ data }) => {
  const message = JSON.parse(data);
  const call = pending.get(message.id);
  if (!call) return;
  pending.delete(message.id); clearTimeout(call.timer);
  if (message.error) call.reject(new Error(JSON.stringify(message.error)));
  else call.resolve(message.result);
};
function rpc(method, params = {}, sessionId) {
  const id = ++sequence;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { pending.delete(id); reject(new Error(`Timeout: ${method}`)); }, 10000);
    pending.set(id, { resolve, reject, timer });
    socket.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
  });
}

try {
  const { targetId } = await rpc("Target.createTarget", { url: "about:blank" });
  const { sessionId } = await rpc("Target.attachToTarget", { targetId, flatten: true });
  const call = (method, params) => rpc(method, params, sessionId);
  const evaluate = async (expression) => {
    const result = await call("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
    if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  };
  const waitFor = async (expression) => {
    for (let attempt = 0; attempt < 100; attempt++) {
      if (await evaluate(expression)) return;
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
    throw new Error(`UI condition not reached: ${expression}`);
  };
  const screenshot = async (filename) => {
    const { data } = await call("Page.captureScreenshot", { format: "png", captureBeyondViewport: false });
    writeFileSync(path.join(output, filename), Buffer.from(data, "base64"));
  };
  await call("Page.enable");
  await call("Emulation.setDeviceMetricsOverride", { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false });
  await call("Page.navigate", { url: base });
  await waitFor("document.querySelector('.document-row input') && !document.getElementById('file-input').disabled");
  await screenshot("client-desktop.png");
  await evaluate(`(() => { document.querySelector('.document-row input').click();
    const input=document.getElementById('question'); input.value='Question de controle interface';
    input.dispatchEvent(new Event('input',{bubbles:true})); document.getElementById('question-form').requestSubmit(); })()`);
  await waitFor("document.querySelector('.answer-content table') && !document.getElementById('question').disabled");
  assert.equal(await evaluate("Boolean(window.webXSS || document.querySelector('#messages img'))"), false);
  await evaluate("document.querySelector('.source').open=true");
  await screenshot("client-answer.png");
  const beforeUpload = await evaluate("document.querySelectorAll('.document-row').length");
  await evaluate(`(() => { const transfer=new DataTransfer();
    transfer.items.add(new File(['interface test'], 'controle-ui.pdf', {type:'application/pdf'}));
    const upload=document.getElementById('file-input'); upload.files=transfer.files;
    upload.dispatchEvent(new Event('change',{bubbles:true})); })()`);
  await waitFor(`document.querySelectorAll('.document-row').length > ${beforeUpload} && !document.getElementById('file-input').disabled`);
  assert.equal(await evaluate("document.getElementById('notice').classList.contains('success')"), true);
  await call("Emulation.setDeviceMetricsOverride", { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  await evaluate(`(() => { const input=document.getElementById('question'); input.value='Question mobile de controle';
    input.dispatchEvent(new Event('input',{bubbles:true})); document.getElementById('question-form').requestSubmit(); })()`);
  await waitFor("document.querySelectorAll('.answer-content table').length === 2 && !document.getElementById('question').disabled");
  assert.equal(await evaluate("document.documentElement.scrollWidth <= window.innerWidth+1"), true, "Mobile page overflows horizontally.");
  await screenshot("client-mobile.png");
  console.log("Browser smoke passed: document selection, question, safe sources, background upload, desktop/mobile layout.");
} finally {
  await rpc("Browser.close").catch(() => browser.kill());
  socket.close();
}
