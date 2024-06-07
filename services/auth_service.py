
class AuthService:
    def __init__(self,database_connector):
        self.db = database_connector
    
        
    def login(self, jwt_token: str) -> str:
        try:
            decoded_token = self.db.auth.verify_id_token(jwt_token)
            uuid = decoded_token['user_id']
            return uuid
        except Exception as e:
            return None

    def logout(self, jwt_token: str) -> bool:
        try:
            decoded_token = self.db.auth.verify_id_token(jwt_token)
            uuid = decoded_token['user_id']
            self.db.auth.delete_user(uuid) #??
            return True
        except Exception as e:
            return False