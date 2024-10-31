for hour in range(0, 24):
    print(f"Hour: {hour} \n")
    for index, model in enumerate(self.models.values()):
        print(f"Trying to predict\n")
        model_name = self.model_keys[index]
        print(f"Predictions of {model_name}")
        if isinstance(model, Sequential):  # For Keras model
            print("LSTM start")
            # Reshape last_days_data for LSTM input
            last_days_data_reshaped = last_days_data.values.reshape(1, last_days_data.shape[0], last_days_data.shape[1])
            print(f"LAST DATA RESHAPED LSTM: {last_days_data_reshaped}")
            raw_prediction = model.predict(last_days_data_reshaped)  # Make predictions
            prediction = raw_prediction.flatten()  # Flatten array if needed
            print(f"LSTM flat predition: {prediction}")
            predictions[model_name].append(prediction[0])
            print("LSTM middle")
            # latest_input = last_days_data_reshaped.iloc[1:]
            # # Append the new prediction to the input DataFrame
            # new_row = pd.DataFrame([latest_input], columns=latest_input.columns)  # Make sure the column names match
            # latest_input = pd.concat([latest_input, new_row], ignore_index=True)

            # Prepare new input
            # last_days_data_reshaped = np.append(last_days_data_reshaped[:, 1:, :],prediction.reshape(1, 1, -1), axis=1)

            new_row = pd.DataFrame([last_days_data.iloc[-1].values], columns=last_days_data.columns)
            new_row.iloc[0, -1] = prediction[0]  # Update last value with prediction
            print("LSTM end")
            last_days_data = pd.concat([last_days_data.iloc[1:], new_row], ignore_index=True)
            print("LSTM finished")
        else:
            print("Beggining")
            # prediction = model.predict(last_days_data.values)  # Make predictions
            prediction = model.predict(last_days_data.iloc[-1:].values)
            predictions[model_name].append(prediction[0])

            # latest_input = last_days_data.iloc[1:]
            # # Append the new prediction to the input DataFrame
            # new_row = pd.DataFrame([latest_input], columns=latest_input.columns)  # Make sure the column names match
            # latest_input = pd.concat([latest_input, new_row], ignore_index=True)

            print("Middle")
            # Prepare new input (for non-Keras models)
            # last_days_data = last_days_data.iloc[1:]
            # new_row = pd.DataFrame([prediction],columns=last_days_data.columns)# Ensure column names match

            new_row = pd.DataFrame([last_days_data.iloc[-1].values], columns=last_days_data.columns)
            new_row.iloc[0, -1] = prediction[0]  # Update last value with prediction

            print("End")
            # last_days_data = pd.concat([last_days_data, new_row], ignore_index=True)
            last_days_data = pd.concat([last_days_data.iloc[1:], new_row], ignore_index=True)
            print("Finished")
        # predictions[model_name] = prediction  # Store predictions by model name
        print(f"Predictions from {model_name}: {prediction}")