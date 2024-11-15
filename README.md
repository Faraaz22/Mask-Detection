# Face Mask Detection using CNN

## Dataset

The dataset used contains two categories:
- **With Mask**: Images of people wearing face masks.
- **Without Mask**: Images of people not wearing face masks.

The dataset consists of **7553** images, with **3725** images of people wearing masks and **3828** images of people not wearing masks.

Dataset Link: [Face Mask Dataset on Kaggle](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)

## Project Structure

```
├── data/
│   ├── with_mask/
│   └── without_mask/
├── uploads/
    ├──uploaded images
├── app.py
├── static/
    ├──styles.css
├── templates/
    ├──index.css                # Contains the model definition and training code
├── FaceMaskDetection(1).ipynb       # Jupyter notebook to train the model
├── mask_model.h5           # Saved trained model
├── requirements.txt        # List of required dependencies
└── README.md               # Project documentation
```


You can install the dependencies using `pip`:
```bash
pip install -r requirements.txt
```
## Running the Flask Application

```bash
flask run
```
Open in Browser:

Navigate to http://127.0.0.1:5000/ in your browser. You should see the homepage with a form to upload an image.
## Model Architecture

The model uses a simple CNN architecture with the following layers:
1. **Conv2D Layer**: Convolutional layer with 16 filters (3x3 kernel size), ReLU activation.
2. **MaxPooling2D Layer**: Max pooling layer with 2x2 pool size.
3. **Conv2D Layer**: Convolutional layer with 32 filters (3x3 kernel size), ReLU activation.
4. **MaxPooling2D Layer**: Max pooling layer with 2x2 pool size.
5. **Flatten Layer**: Flatten the 3D tensor into a 1D vector.
6. **Dense Layer**: Fully connected layer with 64 neurons and ReLU activation.
7. **Dropout Layer**: Dropout layer with a 20% drop rate to prevent overfitting.
8. **Dense Layer**: Fully connected layer with 32 neurons and ReLU activation.
9. **Dropout Layer**: Another dropout layer with a 20% drop rate.
10. **Dense Layer**: Final output layer with 2 neurons (for binary classification: with mask vs without mask), using softmax activation.

The model is compiled using:
- **Optimizer**: Adam
- **Loss Function**: Sparse categorical cross-entropy
- **Metric**: Accuracy

## Training

The model is trained using the following code snippet:

```python
history = model.fit(X_train_scaled, y_train, epochs=10, validation_split=0.1, callbacks=[early_stopping])
```

### Early Stopping
Early stopping is used to prevent overfitting. It monitors the validation loss and stops training if the validation loss doesn't improve after 3 consecutive epochs.

## Model Evaluation

After training, the model is evaluated on the test set to determine its accuracy:

```python
loss, score = model.evaluate(X_test_scaled, y_test, verbose=0)
print("Test accuracy:", score)
```

The model achieved a test accuracy of approximately **92.5%**.

## Model Prediction

You can use the trained model to predict whether a person in a given image is wearing a mask or not. Example code to make predictions:

```python
input_image_path = 'path/to/your/image.jpg'
input_image = cv2.imread(input_image_path)
input_image_resized = cv2.resize(input_image, (128, 128))
input_image_scaled = input_image_resized / 255
input_image_reshaped = np.reshape(input_image_scaled, [1, 128, 128, 3])

input_prediction = model.predict(input_image_reshaped)
input_pred_label = np.argmax(input_prediction)

if input_pred_label == 1:
    print('The person in the image is wearing a mask')
else:
    print('The person in the image is not wearing a mask')
```

## Save the Model

After training, the model is saved to a file using:

```python
model.save('mask_model.h5')
```

## Results

During training, the model's accuracy and loss are plotted:

- **Training Loss vs Validation Loss**
- **Training Accuracy vs Validation Accuracy**

These plots help in evaluating the model's performance over time.

## Screenshots
<img width="953" alt="image" src="https://github.com/user-attachments/assets/566b863f-61db-4679-ab06-fc6fadeff6b1">
<img width="950" alt="image" src="https://github.com/user-attachments/assets/65762d7b-264b-4cd3-afa9-9610cb41ea38">
<img width="938" alt="image" src="https://github.com/user-attachments/assets/041e6b98-2c23-41c5-a3db-d88f12559456">

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



