
import requests, zipfile, io
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import itertools
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
import sys
url_courses=os.environ['URL_COURSES']
url_configs=os.environ['URL_CONFIGS']
url_post=os.environ['URL_POST']
import logging
logger = logging.getLogger(__name__)
def download_data():
    logger.warning("Загружаю данные о курсах")
    r= requests.get(url_courses)
    if r.status_code == 200:
        logger.warning("===> Данных о курсах успешно загружены")
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall("data/")
    else:
        logger.warning("===> Загрузите данные о курсах на портал")
        sys.exit()
def download_configs():
    logger.info("Загружаю файлы конфигурации")
    r= requests.get(url_configs)
    if r.status_code == 200:
        logger.warning("===> Файлы конфигурации успешно загружены")
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall("configs/")
       
    else:
        logger.warning("===> Загрузите данные о конфирации на портал")
        sys.exit()

def post_data(data):
    r= requests.post(url_post, data=json.dumps(data), headers={
    'Content-type':'application/json', 
    'Accept':'application/json'})
    if r.status_code == 200:
         logger.warning("===> Данных о рекомендация загружены на портал")
    else:
        logger.warning("===> Не удалось загрузить рекомендации на портал:")
        logger.warning(r.json()["message"])
        sys.exit()    

def df_from_jsons(path_to_json = 'data/'):
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
    
    data=pd.DataFrame([],columns=["ID курса","Описание курса","Навыки\тэги",'Содержимое'])
    for single_file in json_files:
        with open(path_to_json+single_file, 'r') as f:
            json_file = json.load(f)         
            data=pd.concat([data,pd.DataFrame(data={'ID курса': json_file['id'],'Описание курса':json_file['description'],'Навыки\тэги':json_file['skills'],'Содержимое':json_file['content']},index=[0])])

    return data
    
def run_model():
    download_data()
    download_configs()
    infernce_model=Recomedation()
    post_data(infernce_model.get_model_recomendation_all())


class Recomedation:
    def __init__(self):
        self.queryes=pd.read_pickle('src/feature.pkl')
        self.model=SentenceTransformer("src/fine_tuned_model")
        self.courses=df_from_jsons(path_to_json = 'data/')
        self.dict_profeesional_roles=self.download_data('configs/dict_profeesional_roles')
        self.dict_level_position=self.download_data('configs/dict_level_position')
        self.memory=pd.read_pickle('src/memory.pkl')
    def download_data(self,cname: str):
        return pd.read_csv(cname+'.csv', sep=';',encoding='IBM866')
    def download_pkl(self,cname: str):
        return pd.read_pickle(cname+'.pkl')
    def make_query(self,profession,level):
        if profession in self.queryes.professia.unique():
            #exp =self.queryes[(self.queryes.professia==profession) & (self.queryes.level==level)].exp.values[0]
            skill_set = self.queryes[(self.queryes.professia==profession) & (self.queryes.level==level)].skill_set.values[0]
            query=' '.join(skill_set.split('|'))
            return query
        else:
            return profession
    def infrence_model(self,query):
        sentences_a1 = [query]+[str(row) for row in self.courses['Описание курса'][:]]
        sentences_embeddings_a1 = self.model.encode(sentences_a1)
        #let's calculate cosine similarity for sentence 0:
        similar_value_a1 = cosine_similarity(
            [sentences_embeddings_a1[0]],
            sentences_embeddings_a1[1:])
        return similar_value_a1.tolist()[0]  
    def get_model_recomendation_by_level_position(self,profession,level):
        query=self.make_query(profession,level)  
        return self.infrence_model(query)     
    def  get_model_recomendation_all(self):
        res_list=[]
        tmp=self.courses[:].copy()
        for i,row in self.dict_level_position[:].iterrows():
            splt=str(row.profession_role)+'|'+str(row.level)
            logger.warning(f" Рассчитваю рекомендации для {splt} \n")

            tmp['est']=self.get_model_recomendation_by_level_position(row.profession_role,row.level)
            res_dict={"type":"career","subjectArea":row.profession_role,"level": row.level, 
                "courses":tmp.sort_values(by='est',ascending=0)['ID курса'].values[:30].tolist()}
            res_list.append(res_dict)
        logger.warning(res_list)
        return res_list
if __name__ == "__main__":   
    logging.basicConfig(format='%(levelname)s:%(asctime)s %(funcName)s - %(message)s', level=logging.INFO,
                        filename="src/log.txt")
    try:
        logger.warning("SCRIPT STARTED")
        run_model()
        logger.warning("SCRIPT FINISHED SUCCESSFULLY")
    except Exception as e:
        logger.critical(e, exc_info=True)