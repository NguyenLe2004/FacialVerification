import base64
from PIL import Image
from io import BytesIO
import numpy as np

def get_user_encode_from_db(cursor, user_name: str) -> list:
    """
    Retrieve user's face encoding from the database
    """
    sql = """
        SELECT face_encoded 
        FROM user_accounts
        WHERE user_names = ?
    """
    cursor.execute(sql, (user_name,))
    result = cursor.fetchone()
    if result:
        encoded_user_img_bytes = result.face_encoded
        encoded_user_img = np.frombuffer(encoded_user_img_bytes, dtype=np.float32)
        return encoded_user_img.tolist()
    return []

def upload_user_data(cursor, connection, user: str, encoded_user_img: list) -> dict:
    """
    Upload or update user's face encoding in the database
    """
    encoded_user_img_bytes = np.array(encoded_user_img, dtype=np.float32).tobytes()
    
    # Check if user already exists
    select_sql = """
        SELECT 1 FROM user_accounts WHERE user_names = ?
    """
    cursor.execute(select_sql, (user,))
    result = cursor.fetchone()
    
    if result:
        # User already exists
        return {"success": False, "message": f"User '{user}' already exists in the database."}, 400
    # User does not exist, insert new record
    insert_sql = """
        INSERT INTO user_accounts (user_names, face_encoded) VALUES (?, ?)
    """
    cursor.execute(insert_sql, (user, encoded_user_img_bytes))
    
    connection.commit()
    return {"success": True, "message": "User data uploaded successfully."}

def read_image(image) -> Image:
    """
    Read an image from bytes
    """
    image_bytes = base64.b64decode(image.split(',')[-1])
    image = Image.open(BytesIO(image_bytes))
    return image