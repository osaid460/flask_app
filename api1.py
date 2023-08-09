import io
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template
import base64

app = Flask(__name__)

# Load the pretrained model
model = torch.load("mohsind_res34.pt", map_location=torch.device('cpu'))
model.eval()

# Define class labels (adjust these according to your dataset)
class_labels = ['Daiatsu_Core', 'Daiatsu_Hijet', 'Daiatsu_Mira', 'FAW_V2', 'FAW_XPV', 'Honda_BRV', 'Honda_city_1994', 'Honda_city_2000', 'Honda_City_aspire', 'Honda_civic_1994', 'Honda_civic_2005', 'Honda_civic_2007', 'Honda_civic_2015', 'Honda_civic_2018', 'Honda_Grace', 'Honda_Vezell', 'KIA_Sportage', 'Suzuki_alto_2007', 'Suzuki_alto_2019', 'Suzuki_alto_japan_2010', 'Suzuki_carry', 'Suzuki_cultus_2018', 'Suzuki_cultus_2019', 'Suzuki_Every', 'Suzuki_highroof', 'Suzuki_kyber', 'Suzuki_liana', 'Suzuki_margala', 'Suzuki_Mehran', 'Suzuki_swift', 'Suzuki_wagonR_2015', 'Toyota HIACE 2000', 'Toyota_Aqua', 'Toyota_axio', 'Toyota_corolla_2000', 'Toyota_corolla_2007', 'Toyota_corolla_2011', 'Toyota_corolla_2016', 'Toyota_fortuner', 'Toyota_Hiace_2012', 'Toyota_Landcruser', 'Toyota_Passo', 'Toyota_pirus', 'Toyota_Prado', 'Toyota_premio', 'Toyota_Vigo', 'Toyota_Vitz', 'Toyota_Vitz_2010']


# Define image preprocessing transform
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_idx = torch.max(outputs, 1)
        predicted_class = class_labels[predicted_idx.item()]

    return predicted_class

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     global image_url
#     global predicted_class
#
#     predicted_class = None
#     image_url = None
#
#     if request.method == 'POST':
#         if 'image' not in request.files:
#             return jsonify({'error': 'No file part'})
#
#         image = request.files['image'].read()
#         predicted_class = predict_image(image)
#
#         #PIL.Image.fromarray(tensor)
#         # Encode the image in base64 to include it in the HTML
#         image_url = 'data:image/jpeg;base64,' + base64.b64encode(image).decode()
#
#
#     return render_template('index1.html', predicted_class=predicted_class, image_url=image_url)
#

# @app.route('/predict', methods=['POST'])
# def predict():
#     print(image_url)
#     return render_template('index1.html', predicted_class=predicted_class, image_url=image_url)




@app.route('/', methods=['GET'])
def index():
    return render_template('index1.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No file part'})

        image = request.files['image'].read()
        predicted_class = predict_image(image)  #**
        #print(image_url)
        return jsonify({'predicted_class': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)})

# def predict():
#     try:
#         image = request.files['image'].read()
#         predicted_class = predict_image(image)
#         return jsonify({'predicted_class': predicted_class})
#     except Exception as e:
#         return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
    #print(image_url)


# import base64
#
# with open("yourfile.ext", "rb") as image_file:
#     encoded_string = base64.b64encode(image_file.read())