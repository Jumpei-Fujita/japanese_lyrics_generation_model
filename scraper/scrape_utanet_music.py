import requests
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path')
args = parser.parse_args()

def eliminate_num(s):
    result = re.sub(r"\D", "", s)
    return s.replace(result, '')

def get_artist(artist_list):
    url = 'https://www.uta-net.com/user/ranking/artist_betu.html'
    res = requests.get(url)
    #うたネットのアーティストリストの取得
    soup = BeautifulSoup(res.text, "html.parser")
    #totalのアーティストトップ50
    for link in soup.find_all('ol', attrs={ 'class': "ranking-ol-table"})[0].find_all('a'):
        artist_list.append({'歌手名':eliminate_num(link.text), 'artist_url':'https://www.uta-net.com'+link.get('href')})
    #weeklyのアーティストトップ50
    #totalで取得したアーティストとは異なるアーティストのみ取得
    for link in soup.find_all('ol', attrs={ 'class': "ranking-ol-table"})[1].find_all('a'):
        artist_dic = {'歌手名':eliminate_num(link.text), 'artist_url':'https://www.uta-net.com'+link.get('href')}
        if artist_dic not in artist_list:
            artist_list.append(artist_dic)
    return artist_list

def get_music_list(artist_list):
    for i in tqdm(range(len(artist_list))):
        #各アーティストの曲一覧ページ
        res = requests.get(artist_list[i]['artist_url'])
        soup = BeautifulSoup(res.text, "html.parser")
        artist_list[i]['music_list'] = []
        for td in soup.find_all('td', attrs={'class':"side td1"}):
            #各曲の曲名とurlの取得
            artist_list[i]['music_list'].append({'曲名':td.text, 'song_url':'https://www.uta-net.com/'+td.find_all('a')[0].get('href')})
    return artist_list

def get_lyrics(artist_list):
    for i in tqdm(range(len(artist_list))):
        for j in range(len(artist_list[i]['music_list'])):
            #各曲の歌詞を取得
            res = requests.get(artist_list[i]['music_list'][j]['song_url'])
            soup = BeautifulSoup(res.text, "html.parser")
            lyrics = soup.find('div', attrs={'id':"kashi_area"}).get_text('<sep>').replace('\u3000', ' ').split('<sep>')
            lyrics = [i for i in lyrics if len(i)!=0]
            artist_list[i]['music_list'][j]['lyrics'] = lyrics
    return artist_list

if __name__ == "__main__":
    artist_list = []
    artist_list = get_artist(artist_list)
    artist_list = get_music_list(artist_list)
    artist_list = get_lyrics(artist_list)
    with open(args.path + "/utanet_dataset.pkl","wb") as f:
        pickle.dump(artist_list, f)

