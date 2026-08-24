[Documentation](https://doc.inspireface.online/using-with/python.html#example-face-recognition-with-featurehub)

Alles im folder python\ tests/inspireface

1. virtual environment starten
```bash
source .venv/bin/activate
```
# Tracking ohne reidentification
(inspireface_test.py)

Also ich habe es mal laufen lassen, zwei wichtige bash commands

2. Dann für live video tracking
```bash
python inspireface_test.py 0 --show
```

3. Und für pre recorded Video:
```bash
python inspireface_test.py TH_Video.mp4 --show --out processed_output.avi
```
Die flag show zeigt den process live, kann man weglassen weil der output dann in der .avi Detei zu finden sein wird.
# Tracking mit reidentification
(inspireface_recognition_test.py)

2. live video
```bash
python inspireface_recognition_test.py 0 --show
```
Um live video abzubrechen, einfach im preview/show window sein und q drücken

3. pre recorded video
```bash
python inspireface_recognition_test.py TH_Video.mp4 --out processed_recognition_output.avi
```