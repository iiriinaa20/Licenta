import firebase_admin
from firebase_admin import credentials, auth

class AuthService:
    def __init__(self,credentials_path: str):
        self.credentials = credentials.Certificate(credentials_path)
        self.auth_app = firebase_admin.initialize_app(self.credentials)
        
    def login(self, jwt_token: str) -> str:
        try:
            decoded_token = auth.verify_id_token(jwt_token)
            uuid = decoded_token['user_id']
            return uuid
        except Exception as e:
            return None

    def logout(self, jwt_token: str) -> bool:
        try:
            decoded_token = auth.verify_id_token(jwt_token)
            uuid = decoded_token['user_id']
            auth.delete_user(uuid) #??
            return True
        except Exception as e:
            return False