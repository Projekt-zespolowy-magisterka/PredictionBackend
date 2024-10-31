for hour in range(24):
    print(f"Hour: {hour}\n")
    for index, model in enumerate(self.models.values()):
        print(f"Trying to predict\n")
        model_name = self.model_keys[index]
        print(f"Predicting with {model_name}")
        if isinstance(model, Sequential):  # For Keras (LSTM) model
            print("LSTM start")
            # Reshape last_days_data for LSTM input
            last_days_data_reshaped = last_days_data.values.reshape(1, last_days_data.shape[0], last_days_data.shape[1])
            raw_prediction = model.predict(last_days_data_reshaped)  # Make prediction
            prediction = raw_prediction.flatten()[0]  # Flatten and get single value

            predictions[model_name].append(prediction)  # Store prediction
            print("LSTM prediction complete")

            # Update last_days_data with new prediction
            new_row = pd.DataFrame([last_days_data.iloc[-1].values], columns=last_days_data.columns)
            new_row.iloc[0, -1] = prediction  # Replace the last column with the predicted value (e.g., "Close")
            last_days_data = pd.concat([last_days_data.iloc[1:], new_row], ignore_index=True)  # Shift the window

        else:  # For scikit-learn models
            print("Beginning prediction with non-LSTM model")
            prediction = model.predict(last_days_data.values)  # Make prediction (single-step forecast)
            prediction_value = prediction[0]  # Extract single predicted value
            predictions[model_name].append(prediction_value)  # Store prediction

            # Update last_days_data with new prediction
            new_row = pd.DataFrame([last_days_data.iloc[-1].values], columns=last_days_data.columns)
            new_row.iloc[0, -1] = prediction_value  # Replace the last column with the predicted value
            last_days_data = pd.concat([last_days_data.iloc[1:], new_row], ignore_index=True)  # Shift the window

        print(
            f"{model_name} prediction complete for hour {hour}: {prediction_value if not isinstance(model, Sequential) else prediction}")

    # Optionally log hourly predictions for each model
    print(f"Hour {hour} predictions: { {key: predictions[key][-1] for key in predictions} }\n")

# After loop, 'predictions' will hold 24 hourly predictions for each model for the next day.