import requests

url = "http://localhost:9696/predict"

# Successful Game
game = {
'appid' : 1621680,
'name' : 'Sword and Fairy 4',
'release_year' : 2021,
'release_date' : 'Jun 17, 2021',
'genres' : 'RPG',
'categories' : 'Single-player;Steam Cloud;Family Sharing',
'price' : 12.99,
'developer' : 'SOFTSTAR TECHNOLOGY(SHANGHAI)',
'publisher' : 'SOFTSTAR ENTERTAINMENT'
}


# Unsuccessful Game
# game = {
# 'appid' : 2883660,
# 'name' : 'Dye The Bunny 2',
# 'release_year' : 2024,
# 'release_date' : 'Apr 30, 2024',
# 'genres' : 'Indie',
# 'categories' : 'Single-player;Steam Achievements;Family Sharing',
# 'price' : 6.99,
# 'developer' : 'Vidas Salavejus',
# 'publisher' : 'Vidas Salavejus'
# }

response = requests.post(url, json=game).json()
for item in response:
    print(f"{item['key']}: {item['value']}")






  
  
    
  
 
              
  
  
