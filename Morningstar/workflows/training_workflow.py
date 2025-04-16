import torch
from model.training.data_loader import load_and_split_data
from model.training.training_script import SimpleModel, MODEL_SAVE_PATH, BATCH_SIZE, EPOCHS, LEARNING_RATE
from model.training.evaluation import generate_evaluation_report
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

def main():
    # Configuration
    train_file_path = 'data/processed/btc_final.parquet'  # Example file path
    test_file_path = 'data/processed/btc_final.parquet'   # Example file path for testing

    # Load and split data
    X, y = load_and_split_data(train_file_path)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    # Create DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model, loss function, and optimizer
    input_size = X_train.shape[1]
    model = SimpleModel(input_size)
    criterion = torch.nn.MSELoss()  # Use appropriate loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, labels.unsqueeze(1)).item()
                all_preds.extend(outputs.squeeze().tolist())
                all_labels.extend(labels.tolist())

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_rmse = mean_squared_error(all_labels, all_preds, squared=False)

        print(f'Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val RMSE: {val_rmse:.4f}')

    # Save the model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f'Model saved to {MODEL_SAVE_PATH}')

    # Evaluate the model
    evaluation_report = generate_evaluation_report(model, test_file_path, is_classification=False)
    print("Evaluation Report:")
    print(evaluation_report)

if __name__ == "__main__":
    main()
