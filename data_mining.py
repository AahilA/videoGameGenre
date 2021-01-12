#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
from bs4 import BeautifulSoup
import requests
import numpy as np
import csv


# In[10]:


#Reading the CSV and including only important cols
df = pd.read_csv("steam_games.csv")
df = df[['url','name','popular_tags','genre']]
df = df.dropna(subset=['popular_tags','genre'])


# In[11]:


df


# In[12]:


#Set of genres and Set of genres to exclude
genreSet = {'Action','RPG'}
notAGenre = {'HTC','Massively Multiplayer','Early Access','Valve','Indie','1980s',
 '2.5D','2D Fighter', '360 Video','3D','3D Platformer','3D Vision','4 Player Local', 'Accounting','Action-Adventure',
 'Addictive','Alternate History','Beautiful','Asymmetric VR','Atmospheric','e-sports', 
 'Building', 
 'Arena Shooter',
 'Bullet Hell',
 'CRPG',  'Base Building',
 'Basketball',
 "Beat 'em up",
 'Blood','Cartoony','Cats',
 'Character Customization',
 'Chess',
 'Choices Matter',
 'Choose Your Own Adventure',
 'Cinematic',
 'Classic',
 'Clicker',
 'Co-op',
 'Co-op Campaign',
 'Colorful',
 'Comedy',
 'Comic Book',
 'Competitive',
 'Conspiracy',
 'Controller',
 'Crafting',
 'Crime',
 'Crowdfunded',
 'Cult Classic',
 'Cute','Dark Comedy',
 'Dark Fantasy',
 'Dark Humor','Demons','Destruction',
 'Detective',
 'Difficult',
 'Dinosaurs',
 'Documentary',
 'Dog',
 'Dragons',
 'Drama','Dungeon Crawler',
 'Dungeons & Dragons',
 'Dystopian','Epic',
 'Episodic',
 'Experience',
 'Experimental', 'FMV', 'Faith', 'Fast-Paced',
 'Feature Film',
 'Female Protagonist','First-Person',
 'Fishing',
 'Flight',
 'Free to Play',
 'Funny', 'GameMaker',
 'Games Workshop',
 'God Game', 'Gothic',
 'Great Soundtrack',
 'Grid-Based Movement',
 'Gun Customization',
 'Hack and Slash', 'Hand-drawn',
 'Hardware',
 'Hex Grid',
 'Hidden Object', 'Horses',
 'Hunting',
 'Illuminati',
 'Interactive Fiction',
 'Inventory Management',
 'Investigation',
 'Isometric',
 'JRPG',
 'Kickstarter',
 'Lemmings',
 'Level Editor',
 'Linear',
 'Local Co-Op',
 'Local Multiplayer',
 'Logic',
 'Loot',
 'Lore-Rich',
 'Lovecraftian',
 'MMORPG','Management',
 'Masterpiece',
 'Match 3', 'Memes',
 'Metroidvania', 'Mod',
 'Moddable',
 'Modern',
 'Mouse only',
 'Movie',
 'Multiplayer',
 'Multiple Endings',
 'Music',
 'Music-Based Procedural Generation', 'Mystery Dungeon',  'Narration',
 'Naval',
 'Ninja',
 'Nudity',
 'Old School',
 'Online Co-Op',
 'Open World',
 'Otome',
 'Parkour',
 'Parody',
 'Perma Death',
 'Philisophical',
 'Photo Editing',
 'Physics', 'Point & Click',
 'Political',
 'Politics',
 'Post-apocalyptic',
 'Procedural Generation',
 'Programming',
 'Psychedelic',
 'Psychological',
 'Psychological Horror', 'Puzzle-Platformer',
 'PvP',
 'Quick-Time Events', 'RPGMaker',
 'RTS',  'Real-Time with Pause',
 'Realistic',
 'Relaxing',
 'Replay Value',
 'Robots',  'Rogue-like',
 'Rogue-lite', 'Runner',
 'Sailing', 'Satire', 'Short',
 'Side Scroller',
 'Silent Protagonist',  'Skateboarding', 'Singleplayer',
 'Software',
 'Software Training',
 'Sokoban',
 'Soundtrack', 'Spectacle fighter',
 'Split Screen',  'Star Wars',
 'Stealth',
 'Steam Machine',
 'Steampunk',
 'Story Rich',  'Strategy RPG',
 'Stylized',
 'Supernatural',
 'Surreal', 'Survival Horror', 'Tactical',
 'Team-Based',
 'Tennis',
 'Text-Based',
 'Third Person',
 'Third-Person Shooter',
 'Thriller',
 'Time Attack',
 'Time Manipulation',
 'Time Travel',
 'Top-Down',
 'Top-Down Shooter',
 'Touch-Friendly',
 'Tower Defense', 'Trading',
 'Trading Card Game',
 'Turn-Based',
 'Turn-Based Combat',
 'Turn-Based Strategy',
 'Turn-Based Tactics',
 'Tutorial',
 'Twin Stick Shooter', 'Underwater', 'Emotional',
 'Unforgiving', 'Family Friendly', 'Gaming',
 'Utilities', 'Voice Control',
 'VR',  'Video Production', 'Minigames',
 'Minimalist', 'NSFW',
 'Violent', 'Score Attack',
 'Sexual Content',
 'Visual Novel', 'Pixel Graphics',
 'Voxel', 'PvE',
 'Walking Simulator','Remake'}


# In[13]:


#Making set of Real Genres based on genre and popular tags of games
badGames = pd.DataFrame()
for index,row in df.iterrows():
    a = row['genre']
    b = set(a.split(',')) - notAGenre
    if len(a) != len(b) and len(b) == 0:
        a = row['popular_tags']
        b = set(a.split(',')) - notAGenre
        if len(a) != len(b) and len(b) == 0:
            #has no genre left
            badGames = badGames.append(row)
    genreSet.update(b)


# In[14]:


#Dropping the bad games
df = pd.concat([df, badGames, badGames]).drop_duplicates(keep=False)


# In[15]:


genreSet


# In[16]:


#Creating mapping for One Hot Encoding later on
genreMap = {}
for index, genre in enumerate(genreSet):
    genreMap[genre] = index


# In[17]:


#Creating one hot encoding for a particular set of genres of a game
def getOHE(genres):
    ohe = np.zeros(len(genreMap))
    for genre in set(genres):
        if genre in genreMap.keys():
            ohe[genreMap[genre]] = 1
    return ohe


# # Instructions for peter
# ### Run all the cells one by one and then wait. This should download all the images and place them in a folder and it should generate a csv in the last cell of the game name and one hot vector. 
# Also if u can think of a faster way to download images (maybe first download all from web then write all of them to disk idk if thats faster) that would be great.

# In[ ]:


#Downloading all the images and creating the CSV Label
gameLabel = {}
for index, row in df.iterrows():

    #Incase you want to start from a later index use this and change the 0 value
    if index < 0:
        continue
    
    if index % 100 == 0:
        print(index/df.shape[0])
    
    URL = row['url']
    genre = row['popular_tags'].split(',') + row['genre'].split(',')
    name = row['name'].replace(" ","_").replace('/','_')

    if len(name) > 200:
        continue

    ohe = getOHE(genre)
    gameLabel[name] = ohe
    
#     print(index)
#     print(URL)
#     print(genre)
#     print(name)
#     print(ohe)

        
    page = requests.get(URL)
    soup = BeautifulSoup(page.content,'html.parser')
    r = soup.find(class_='game_header_image_full')
    try:
        response = requests.get(r.get("src"))
    except:
        print("Error fetching game " + name)
        print(URL)
        print("Deleting from labels")
        if name in gameLabel.keys():
            del gameLabel[name]
        continue
    
    file = open("game_images/"+name+'.png', "wb")
    file.write(response.content)
    file.close()

    file = open('gameLabels.csv', 'a')
    file.write("%s,%s\n"%(name,gameLabel[name]))
    file.close()