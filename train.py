# Init wandb
import wandb
wandb.init(project="wandb_tut")

# Load Dataset
import sklearn.datasets
features, labels = sklearn.datasets.fetch_california_housing(return_X_y=True)

# Split to train and test sets
from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=.1)

# Train Model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=10, min_samples_leaf=10)
model.fit(train_features, train_labels)

# Calculate Performance
train_predictions = model.predict(train_features)
test_predictions = model.predict(test_features)

from sklearn.metrics import r2_score
train_r2 = r2_score(train_labels, train_predictions)
test_r2 = r2_score(test_labels, test_predictions)

print(train_r2)
print(test_r2)

# Log metrics with wandb
wandb.log({"Train r2": train_r2,
           "Test r2": test_r2})

# Save model to wandb
import pickle
pickle.dump(model, open('model.sav', 'wb'))
wandb.save("model.sav")
wandb.save("train.py")

# Log metrics in a loop
for i in range(1000):
    wandb.log({'test_loop_log': i**2})