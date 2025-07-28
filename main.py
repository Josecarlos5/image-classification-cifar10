from src.data_preprocessing import load_data
from src.model import create_model
from src.train import train_model
from src.evaluate import evaluate_model

def main():
    # Load data
    (x_train, y_train), (x_test, y_test) = load_data()
    
    # Create model
    model = create_model()
    
    # Train model
    train_model(model, x_train, y_train, x_test, y_test)
    
    # Evaluate model
    evaluate_model(model, x_test, y_test)

if __name__ == '__main__':
    main()
