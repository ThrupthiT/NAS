def classify_dataset(file_path):
    if os.path.isdir(file_path):
        text_files = [f for f in os.listdir(file_path) if f.lower().endswith('.txt')]
        image_files = [f for f in os.listdir(file_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if text_files:
            return "Text_Dataset"
        elif image_files:
            return "Image_Dataset"
        else:
            return "Unknown"
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension in ['.csv', '.xls', '.xlsx']:
        try:
            data = pd.read_csv(file_path)
            if data.shape[1] > 1:
                return "Tabular_Dataset"
        except Exception:
            return "Unknown"
    elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp']:
        try:
            img = cv2.imread(file_path)
            if img is not None:
                return "Image_Dataset"
        except Exception:
            return "Unknown"
    elif file_extension == '.txt':
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            if content.strip():
                return "Text_Dataset"
        except Exception:
            return "Unknown"
    return "Unknown"
