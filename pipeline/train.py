import os
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '-1'

import random
import tensorflow as tf
import numpy as np
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

from pipeline.split import split_for_training
from pipeline.preprocessing import process
from utils.model_architecture import build_model
from utils.settings import numeric_features, TARGET, categorical_features, MAX_LEN, PAD_VALUE
from utils.model_architecture import build_model, compile_model
from pipeline.visualize_predictions import save_to_pdf
from utils.prediction_handeling import handle_negative_predictions
import joblib
from pathlib import Path
from sklearn.preprocessing import RobustScaler
import pandas as pd


def main():


    input_scaler = RobustScaler()
    output_scaler = RobustScaler()
    
    train_df, test_df = split_for_training()

    input_scaler.fit(train_df[numeric_features])
    output_scaler.fit(train_df[[TARGET]])

    X_train, y_train, mask_train = process(train_df, input_scaler, output_scaler)
    X_test, y_test, mask_test = process(test_df, input_scaler, output_scaler)
    
    n_features = X_train.shape[2]

    model = build_model(input_shape=(MAX_LEN, n_features), PAD_VALUE=PAD_VALUE)
    model, early_stopping, reduce_lr = compile_model(model)

    model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=15,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1,
    shuffle=False
    )

    y_pred_scaled = model.predict(X_test)
    y_pred = output_scaler.inverse_transform(y_pred_scaled)
    y_true = output_scaler.inverse_transform(y_test)

    y_pred = handle_negative_predictions(y_pred)
    placement_ids_to_show = test_df['placement_id'].unique()[:5]

    BASE = Path(__file__).parent.parent
    visualization_path = BASE / "visualizations"


    # df_pred = pd.DataFrame(y_pred)
    # df_pred.to_csv(BASE / "assets" / "y_pred.csv", index_label='sample_index')
    # print("Saved as y_pred.csv")

    save_to_pdf(
    placement_ids_to_show=placement_ids_to_show,
    df_test=test_df,
    model=model,
    input_scaler=input_scaler,
    output_scaler=output_scaler,
    numeric_features=numeric_features,
    categorical_features=categorical_features,
    TARGET=TARGET,
    MAX_LEN=MAX_LEN,
    PAD_VALUE=PAD_VALUE,
    given_days=[0, 7, 14, 21, 28],
    pdf_path=visualization_path / "tahona_daily_loss_predictions.pdf"
    )

    model.save(BASE / "assets" / "model.keras")
    joblib.dump(input_scaler, BASE / "assets" / "input_scaler.joblib")
    joblib.dump(output_scaler, BASE / "assets" / "output_scaler.joblib")

if __name__ == "__main__":
    main()
