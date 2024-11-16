# Face Mask Detection using CNN

The project utilizes the concept of CNNs to to make predcitions based on given image input whether the person in the image is wearing a face mask or not.

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
    ├──index.css                
├── FaceMaskDetection(1).ipynb      
├── mask_model.h5         
├── requirements.txt        
└── README.md              
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

## Screenshots
<img width="953" alt="image" src="https://github.com/user-attachments/assets/566b863f-61db-4679-ab06-fc6fadeff6b1">
<img width="950" alt="image" src="https://github.com/user-attachments/assets/65762d7b-264b-4cd3-afa9-9610cb41ea38">
<img width="938" alt="image" src="https://github.com/user-attachments/assets/041e6b98-2c23-41c5-a3db-d88f12559456">

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



