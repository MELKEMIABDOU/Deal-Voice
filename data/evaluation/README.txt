JEU D’ÉVALUATION — 50 QUESTIONS EN FRANÇAIS

Contenu : questions.jsonl, verification.json, et les trois documents sources originaux.
Distribution : 12 audio_fact (transcriptions), 8 pdf_text, 15 table_*, 5 multi_*, 10 unanswerable.

AUDIO : B2B-Appel.txt et Reunion-dequipe.txt sont les références textuelles des 12 questions audio. Les fichiers B2B Appel.mp3 et Reunion dequipe.mp3 sont aussi présents pour l’ingestion ; ces questions évaluent les faits des transcriptions, pas la reconnaissance vocale, la diarisation acoustique ni les horodatages. Pour évaluer l’ASR, comparer séparément les transcriptions produites aux fichiers TXT de référence.

CITATIONS : pages physiques du PDF (couverture = page 1). Les intitulés table correspondent aux titres de section ou d’annexe ; section identifie les passages narratifs. Pour les TXT sans pages ni horodatages, seule la référence fichier est fournie.

RÉPONSES ABSENTES : expected_behavior=refuse signifie reconnaître que l’information n’est pas fournie, pas opposer un refus de sécurité. expected_answer reste null conformément au format demandé. Les contrôles d’absence et les formulations acceptables sont consignés dans verification.json.

CALCULS : les sommes et différences sont vérifiées ; les colonnes calculées sont identifiées explicitement. Les anomalies du PDF sont conservées dans les sources et traitées dans les questions, jamais corrigées silencieusement.

VALIDATION : 50 objets JSON lisibles, un par ligne, identifiants uniques, langue fr, distribution exacte, références aux seuls fichiers fournis, pages PDF entre 1 et 7.
