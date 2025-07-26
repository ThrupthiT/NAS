import numpy as np
def search_and_select_best_model(dataset_type):
    # If the model is already saved, load it and return
    model_path = f"best_model_{dataset_type}.h5"
    if os.path.exists(model_path):
        print(f"Loading pre-trained model for {dataset_type}...")
        model = load_model(model_path)
        return model, model.summary()
    if dataset_type == "Tabular_Dataset":
        # Load the dataset directly here
        df = pd.read_csv(file_path)
        tabular_task = detect_tabular_task(df)
        print(f"Tabular Task: {tabular_task}")
        data = df.iloc[:, :-1].values  # Features (everything except the last column)
        labels = df.iloc[:, -1].values  # Target (the last column)
        # For Tabular Dataset, use Keras Tuner for NAS
        print("Searching for the best model for Tabular Dataset...")
        def build_model(hp):
            model = tf.keras.Sequential()
            model.add(layers.InputLayer(input_shape=(data.shape[1],)))
            model.add(layers.Dense(hp.Int('units', min_value=32, max_value=256, step=32), activation='relu'))
            model.add(layers.Dense(1, activation='sigmoid' if tabular_task == "Classification" else 'linear'))
            model.compile(optimizer='adam', 
                          loss='binary_crossentropy' if tabular_task == "Classification" else 'mean_squared_error', 
                          metrics=['accuracy'])
            return model
        tuner = kt.Hyperband(build_model, objective='val_accuracy', max_epochs=10, hyperband_iterations=2)
        tuner.search(data, labels, epochs=10, validation_split=0.2)
        best_model = tuner.get_best_models(num_models=1)[0]
        best_model.save(model_path)
        return best_model, best_model.summary()
    elif dataset_type == "Image_Dataset":
        # Load the image dataset here (for simplicity, assume it's pre-processed)
        # You can use any image loading technique (e.g., Keras ImageDataGenerator)
        print("Searching for the best model for Image Dataset...")
        # In a real scenario, you should load and preprocess images properly
        image_model = ak.ImageClassifier(max_trials=5)  # Search for 5 models
        image_model.fit(data, labels, epochs=10)
        best_model = image_model.export_model()
        best_model.save(model_path)
        return best_model, best_model.summary()
    if dataset_type == 'Text_Dataset':
        data = np.random.randint(1,10000,size=(1000,100))  # Your tokenized text data as padded sequences
        labels = np.random.randint(0,2,size=(1000,))  # Corresponding labels (0/1)
        text_model = tf.keras.Sequential([
            layers.Embedding(input_dim=10000, output_dim=64),  # No need for input_length now
            layers.LSTM(128, return_sequences=True),
            layers.LSTM(64),
            layers.Dense(1, activation='sigmoid')
        ])
        text_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        text_model.fit(data, labels, epochs=10, validation_split=0.2)
        text_model.save(model_path)
        return text_model, text_model.summary()

    else:
        raise ValueError("Unsupported dataset type")