# "Not Apple" training images

Photos placed in this folder teach the classifier to say **"this is not an
apple"** instead of forcing a ripeness label onto leaves, hands, branches and
other things the detector mistakes for fruit.

The folder is optional. With no images here, `train_final_model.py` trains the
usual six ripeness classes and the app behaves exactly as before.

## What to add

Roughly **100 or more** `.jpg` / `.jpeg` / `.png` images (the same count as one
apple class). Variety matters more than quantity:

- Leaves: green, yellow, dark, wet, single and in clusters, close-up and from a distance
- Branches, stems and bark
- Hands and fingers holding nothing, or holding something that is not an apple
- Empty backgrounds you photograph apples against (table, tray, grass, hand)
- Other round or green/red/yellow objects (other fruit, balls, bottle caps)

Best of all: photos taken **in the same way as your apple photos** (same
camera, lighting and background) with the apple left out. The model learns
fastest from the cases it will actually be confused by.

Do not put apples in here, including rotten or unusual ones. Those belong in
the ripeness folders.

## Train

From the repository root:

```bash
python "Model Training/train_final_model.py"
```

The script prints a cross-validated check: how many non-apples are correctly
rejected, and how many real apples are wrongly rejected. Aim for the second
number to stay under about 3%. It then overwrites
`Model Training/svm_final_model.pkl`, which the app loads.

Train with the same scikit-learn version the rest of the team uses; a model
pickled by a different version triggers warnings and may fail to load.
