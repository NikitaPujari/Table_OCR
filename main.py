from fastapi import FastAPI, Request, Body
import cv2
import pytesseract
import pandas as pd
from os import listdir
import base64
import io
import json
import numpy as np
from PIL import Image
import os
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional
from passlib.context import CryptContext
from datetime import datetime, timedelta
import jwt
import uvicorn

app = FastAPI()

#Define your JWT-related configurations
SECRET_KEY = "your-secret-key"  # Replace with your secret key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

#Create a User model to represent user credentials (username and password)
class User(BaseModel):
    username: str
    password: str

#Create a Token model to represent the JWT token
class Token(BaseModel):
    access_token: str
    token_type: str

#Create a Security instance to handle JWT authentication
security = HTTPBearer()

#Create a pwd_context instance to handle password hashing and verification
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

#Create a get_password_hash function to hash passwords
def get_password_hash(password: str):
    return pwd_context.hash(password)

#Create a verify_password function to verify passwords
def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)

#Create a create_access_token function to generate JWT tokens
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


@app.get('/')
async def hello_world():
    return 'Hello World'

#API endpoints with JWT authentication
@app.post("/login")
async def login(user: User):
    # Check credentials here, e.g., validate username and password against a database
    # Replace this with your actual login logic

    # Example login logic:
    if user.username == "admin" and user.password == "password":
        # Generate JWT token
        access_token = create_access_token(
            data={"sub": user.username},
            expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        )
        return Token(access_token=access_token, token_type="bearer")

    # If credentials are invalid, raise an HTTPException
    raise HTTPException(status_code=401, detail="Invalid username or password")

@app.post("/table_ocr")
async def table_ocr(request=Body(), credentials: HTTPAuthorizationCredentials = Depends(security)):
#async def table_ocr(request=Body()):
    try:
        # Verify the token and extract the payload
        payload = jwt.decode(
            credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM]
        )
        data = request
        #data = await request.json()
        encoded_string = data["base64"]
        
        imgdata = base64.b64decode(encoded_string)
        nparr = np.frombuffer(imgdata, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        height, width, channels = img.shape
        print(height)
        print(width)
        
        with open("obj.names", 'r') as f:
            classes = f.read().splitlines()
        
        net = cv2.dnn.readNetFromDarknet('yolov4-obj.cfg', 'yolov4-obj_final.weights')
        model = cv2.dnn_DetectionModel(net)
        model.setInputParams(scale=1/255, size=(416, 416), swapRB=True)
        classIds, scores, boxes = model.detect(img, confThreshold=0.6, nmsThreshold=0.4)
        
        header_count = 0
        column_count = 0
        table_count = 0
        row_count = 0
        print_count = 0
        
        for (classId, score, box) in zip(classIds, scores, boxes):
            x, y, w, h = box
            crop_img = img[y:y+h, x:x+w]
            class_name = classes[classId]
            
            if class_name == 'Header':
                if x < 100:
                    header_count = 1
                elif x > 100 and x < 400:
                    header_count = 2
                elif x > 400 and x < 500:
                    header_count = 3
                elif x > 500 and x < 700:
                    header_count = 4
                print_count = header_count
            elif class_name == 'Row':
                if y < 450:
                    row_count = 1
                else:
                    row_count = 2
                print_count = row_count
            elif class_name == 'Column':
                if x < 100:
                    column_count = 1
                elif x > 100 and x < 400:
                    column_count = 2
                elif x > 400 and x < 500:
                    column_count = 3
                elif x > 500 and x < 700:
                    column_count = 4
                print_count = column_count
            elif class_name == 'Table':
                table_count += 1
                print_count = table_count
            
            image_name = class_name + '_' + str(print_count)
            print(image_name)
            
            cv2.imwrite('Cropped_image_folder/cropped_image_{}.png'.format(image_name), crop_img)
            
            #For windows
            #pytesseract.pytesseract.tesseract_cmd = os.path.join(os.getcwd(), 'Tesseract-OCR/tesseract.exe')
            files = [f for f in listdir('Cropped_image_folder')]
            image_to_text_dict = {}
            
            for file in files:
                #image = cv2.imread(os.path.join('Cropped_image_folder', file))
                file_name = os.path.join('Cropped_image_folder', file)
                print("####################################################")
                print(file_name)
                image_to_text = pytesseract.image_to_string(os.path.join('Cropped_image_folder', file))
                #image_to_text = pytesseract.image_to_string(image)
                image_to_text_stripped = image_to_text.rstrip("\n\x0c")
                image_to_text_dict[file.split('.')[0].split('_')[2] + '_' + file.split('.')[0].split('_')[3]] = image_to_text_stripped
        
        return image_to_text_dict
    
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")

    except jwt.DecodeError:
        raise HTTPException(status_code=401, detail="Could not decode token")

    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=5000)
