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

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages



def visualize_predictions_given_days_new_masked(
    placement_ids_to_show, df_test, model,
    input_scaler, output_scaler,
    numeric_features, categorical_features,
    TARGET, MAX_LEN, PAD_VALUE,
    mask_test=None,
    given_days=[0, 10, 15, 20, 30]
):
    os.environ['PYTHONHASHSEED'] = '42'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    error_all = []

    for placement_id_to_show in placement_ids_to_show:
        placement_data = (
            df_test[df_test['placement_id'] == placement_id_to_show]
            .sort_values('age')
        )

        if len(placement_data) == 0:
            print(f"Placement {placement_id_to_show} not found in test set")
            continue

        true_loss = placement_data[TARGET].values
        n_days = len(true_loss)

        # Scale features
        numeric_scaled = input_scaler.transform(
            placement_data[numeric_features]
        )
        categorical_vals = placement_data[categorical_features].values
        if categorical_vals.ndim == 1:
            categorical_vals = categorical_vals.reshape(-1, 1)

        X_full = np.concatenate(
            [numeric_scaled, categorical_vals], axis=1
        )

        error = []

        for given_day in given_days:
            if given_day >= n_days - 1:
                error.append(np.nan)
                continue


            seq_len = min(given_day + 1, n_days)
            X_input = X_full[:seq_len]

            if seq_len < MAX_LEN:
                pad = np.full(
                    (MAX_LEN - seq_len, X_full.shape[1]),
                    PAD_VALUE
                )
                X_input = np.vstack([pad, X_input])

            X_input = X_input.reshape(1, MAX_LEN, X_full.shape[1])

            y_pred_scaled = model.predict(X_input, verbose=0)
            y_pred = output_scaler.inverse_transform(y_pred_scaled)
            y_pred = y_pred.reshape(-1)

            future_start = given_day + 1

            true_future = true_loss[future_start:]
            pred_future = y_pred[future_start:future_start + len(true_future)]

            cum_true_future = np.sum(true_future)
            cum_pred_future = np.sum(pred_future)

            diff = cum_true_future - cum_pred_future
            error.append(diff)

            plt.figure(figsize=(12, 6))
            plt.plot(true_loss, label="True")
            plt.plot(y_pred, label="Pred")

            plt.axvline(
                x=future_start - 1,
                color="red",
                linestyle="--",
                label="Prediction start"
            )
            
            plt.title(
                f"Placement {placement_id_to_show} | Given days: {given_day}\n"
                f"Future true sum: {int(cum_true_future)}, "
                f"Future pred sum: {int(cum_pred_future)}, "
                f"Diff: {int(diff)}"
            )

            plt.xlabel("Day")
            plt.ylabel("Daily Loss")
            plt.legend()
            plt.show()

        error_all.append(error)

    return error_all


def save_to_pdf(
    placement_ids_to_show, df_test, model,
    input_scaler, output_scaler,
    numeric_features, categorical_features,
    TARGET, MAX_LEN, PAD_VALUE,
    mask_test=None,
    given_days=[0, 10, 15, 20, 30],
    pdf_path="predictions.pdf"
):
    """
    Visualize and save predictions for multiple placements in a PDF.
    Only calculates cumulative difference for future days (beyond given_day).
    """
    error_all = []

    with PdfPages(pdf_path) as pdf:
        for placement_id_to_show in placement_ids_to_show:
            placement_data = (
                df_test[df_test['placement_id'] == placement_id_to_show]
                .sort_values('age')
            )

            if len(placement_data) == 0:
                print(f"Placement {placement_id_to_show} not found in test set")
                continue

            true_loss = placement_data[TARGET].values
            n_days = len(true_loss)

            # Scale numeric features
            numeric_scaled = input_scaler.transform(
                placement_data[numeric_features]
            )

            # Combine with categorical features
            categorical_vals = placement_data[categorical_features].values
            if categorical_vals.ndim == 1:
                categorical_vals = categorical_vals.reshape(-1, 1)

            X_full = np.concatenate([numeric_scaled, categorical_vals], axis=1)

            error = []

            for given_day in given_days:
                if given_day >= n_days - 1:
                    error.append(np.nan)
                    continue

                # ------------------------
                # Build input
                # ------------------------
                seq_len = min(given_day + 1, n_days)
                X_input = X_full[:seq_len]

                if seq_len < MAX_LEN:
                    pad = np.full(
                        (MAX_LEN - seq_len, X_full.shape[1]),
                        PAD_VALUE
                    )
                    X_input = np.vstack([pad, X_input])

                X_input = X_input.reshape(1, MAX_LEN, X_full.shape[1])

                # ------------------------
                # Predict
                # ------------------------
                y_pred_scaled = model.predict(X_input, verbose=0)
                y_pred = output_scaler.inverse_transform(y_pred_scaled).flatten()
                y_pred = np.round(y_pred).astype(int)  # Round to integer counts

                # ------------------------
                # FUTURE-ONLY difference
                # ------------------------
                future_start = given_day + 1
                true_future = true_loss[future_start:]
                pred_future = y_pred[future_start:future_start + len(true_future)]

                cum_true_future = np.sum(true_future)
                cum_pred_future = np.sum(pred_future)
                diff = cum_true_future - cum_pred_future
                error.append(diff)

                # ------------------------
                # Plot
                # ------------------------
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(true_loss, label="True")
                ax.plot(y_pred, label="Pred")
                ax.axvline(
                    x=future_start - 1,
                    color="red",
                    linestyle="--",
                    label="Prediction start"
                )

                ax.set_title(
                    f"Placement {placement_id_to_show} | Given days: {given_day}\n"
                    f"Future true sum: {cum_true_future}, "
                    f"Future pred sum: {cum_pred_future}, "
                    f"Diff: {diff}"
                )
                ax.set_xlabel("Day")
                ax.set_ylabel("Daily Loss")
                ax.legend()
                ax.grid(True)

                pdf.savefig(fig)  # Save current figure to PDF
                plt.close(fig)    # Close the figure to free memory

            error_all.append(error)

    print(f"All plots saved to {pdf_path}")
    return error_all
