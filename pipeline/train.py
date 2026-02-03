from pipeline.split import split_for_training
from pipeline.preprocessing import process
from utils.model_architecture import build_model
from assets.model_scaler_loader import BASE
from utils.settings import numeric_features, TARGET
from utils.model_architecture import build_model, compile_model
import joblib
from sklearn.preprocessing import RobustScaler

def main():
    train_df, test_df = split_for_training()

    input_scaler = RobustScaler()
    output_scaler = RobustScaler()

    input_scaler.fit(train_df[numeric_features])
    output_scaler.fit(train_df[[TARGET]])
    X_train, y_train = process(train_df, input_scaler)
    # X_test, y_test = process(test_df, input_scaler)
    
    model = build_model()
    model, early_stopping, reduce_lr = compile_model(model)

    model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=15,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
    )
    
    model.save(BASE / "model.keras")
    joblib.dump(input_scaler, BASE / "input_scaler.joblib")
    joblib.dump(output_scaler, BASE / "output_scaler.joblib")

if __name__ == "__main__":
    main()
