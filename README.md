# AI Research Intern In Sinovation Venture

<p align="center">
  
  <img src="https://github.com/Johnny-liqiang/AI-Research-Intern-In-Sinovation-Venture/blob/master/project_gpt2_demo/img/AI%20institute.png" width="200" alt="Sinovation Ventures AI Institute">
  
Sinovation Ventures (创新工场) was founded in September 2009 by Dr. Kai-Fu Lee (previously head of Google China and founder of Microsoft Research Asia) with the purpose of galvanizing a new era of successful Chinese entrepreneurs. 
</p>

Sinovation Ventures (创新工场) are a truly differentiated early stage venture capital fund with $1.2 billion in assets under management, in both USD and RMB funds. As a top domestic China entrepreneurship platform, works closely with founders; provides not just venture financing, but also value-added professional services, including in areas such as UI/UX design, product, marketing, recruiting, legal, government relations, and finance, thereby helping start-up companies grow rapidly in their first couple of years of operation.

# Author List 
[Qiang Li](https://www.linkedin.com/in/qiang-li-166362143/)\*, [Xinyu Bai](https://www.linkedin.com/in/xinyu-bai-8b495b180/)\*, [Qiyi Ye](https://www.linkedin.com/in/qiyi-ye-36703018/)\*  [Sinovation Ventures AI Institute](https://www.sinovationventures.com/)\* <br />

**[[GPT-3 Research Report](https://github.com/Johnny-liqiang/AI-Research-Intern-In-Sinovation-Venture/blob/master/Technical%20report%20on%20GPT-3%20and%20its%20applied%20Business%20Scenario.pdf)] [[5 GPT-Chinese Application Video](https://drive.google.com/drive/folders/1Z17fDHR51ICJTBcGROu9beGm03_tAqur?usp=sharing)] [[API Page](https://10.18.103.82/model/)]** <br />

On this AI leading Intern I am responsible for:

-AI cutting-edge technology research, landing tracking according to the given subdivision technology direction; paper reading and conducting engineering experiments, systematic comparison research, and output the modular technology research report;

-Aiming at the research on the countermeasures of artificial intelligence algorithms, realize and evaluate the performance of state-of-the-art algorithms on text data, and design corresponding improvements and preliminary software models.

## GPT enable Auto-Email Reply 
Inspired by the OpenAI GPT-3 API and OthersideAI .Inc, I developed this Chinese version Auto Email Replier, which is capable of generating specific Emails based on given domain, such as meeting invitation, public speech, interview invitation and office daily communication. By receiving three user
inputs: Email recipient, Email subject and Key points, it uses the NLP GPT-2 model initially trained on 3
million Chinese News and Poetry Dataset. It has around 736 ,744 ,960 parameters , 

<p align="center">
  
  <img src="https://github.com/Johnny-liqiang/AI-Research-Intern-In-Sinovation-Venture/blob/master/project_gpt2_demo/1%20GPT-2%20Chinese.png" width="800" alt="GPT2-Chinese @Sinovation_Venture">
  
</p>

I further collected and filtered Enron Email Dataset (after data preprocessing, it includes around 7,800 emails in
train), and trained a new language model from scratch using Transformers and Tokenizers. As we are
training from scratch , we only initialize from a config, not from an existing pre -trained model , in
Sinovation Ventures . The frontend of this project was developed by a traditional frontend framework ,
primarily using Bootstrap and jQuery to achieve responsive web design , and I helped to build the
entire backend Django server. The use of API through Ajax calls is to ensure smooth email generation.
Then I utilized Nginx cooperated with Uwsgi and launched it into the online server. Boto3, Urllib3 and
Huggface transformers are mainly used, and jieba used for tokenizer; synonym, pycorrector package
used for sentence grammar check.

<p align="center">
  <img src="https://github.com/Johnny-liqiang/AI-Research-Intern-In-Sinovation-Venture/blob/master/project_gpt2_demo/4%20E-mail%20Auto-Replier%20powered%20by%20GPT-2.png" width="800" alt="auto Email reply @Sinovation_Venture">
</p>

## GPT enable Commercial Slogans Generation 
This commercial slogan app can automatically generate commercial statements using fine-tune GPT-2 model based on
user given input such as product, brand name and keyword*optional. I trained this model from the previous initial GPT-2
model, including forming a fine-tuning dataset and performing the data processing work. The model will analyze the
user's input and determine associated user action through information extraction and classification. The output will then
be posted to the frontend through API. The GPT-generated slogan is highly innovative and inspiring.

<p align="center">
  
  <img src="https://github.com/Johnny-liqiang/AI-Research-Intern-In-Sinovation-Venture/blob/master/project_gpt2_demo/5%20Commercial%20Slogans%20powered%20by%20GPT-2.png" width="800" alt="GPT2-Slogan @Sinovation_Venture">
  
</p>

## Dataset filtering and how to finetune customer dataset 
I also provide here the code for finetuning your own dataset and give some good website for NLP dataset.

**[[GPT-2 writing novel @ Qiang](https://drive.google.com/file/d/12jjrd8-EycdljEzp2q5J2xEsFbmis8rs/view?usp=sharing)] [[how to finetune on your dataset](https://colab.research.google.com/drive/1MxgTJ4hRt4k7SkKB5nblrIm5i_cMGjWB?usp=sharing)] [[how to make you own dataset](https://drive.google.com/file/d/1UUdJTBOI18qEZNIzGF_7rNb52SBJ4TOn/view?usp=sharing)]** <br />

## Training

### Training Data 
- Download the our Finetune dataset for Commercial advertisement [Commercial Adv. dataset](https://github.com/Johnny-liqiang/AI-Research-Intern-In-Sinovation-Venture/tree/master/project_gpt2_demo/Commercial%20Adv.%20dataset), 
and [ Finetune dataset for Chinese- Enran Email dataset](https://github.com/Johnny-liqiang/AI-Research-Intern-In-Sinovation-Venture/tree/master/project_gpt2_demo/Finetune%20dataset%20for%20Chinese-%20Enran%20Email%20dataset).
- Preprocess each datasets according the [readme](data/coco/readme.md) files.

### Download the pre-trained GPT2-chinese model (2.9 G) has 736744960 parameters
(This model was trained on the 3 million tokens Chineses News and wiki Dataset, around 30G)
```
cd /content/drive/'My Drive'/'sinovation'/'project_gpt2_demo'/

from transformers import Trainer, TrainingArguments, AutoModelWithLMHead 
from modeling_gpt2 import GPT2Config,GPT2LMHeadModel

config = GPT2Config.from_pretrained("/content/drive/My Drive/sinovation/project_gpt2_demo/model")
model = GPT2LMHeadModel.from_pretrained("/content/drive/My Drive/sinovation/project_gpt2_demo/model", config=config)
model.num_parameters()

```

### Training Email based model
- [Setup](#environment-setup) your environment
- From the experiment directory, run
```
pip install requirement.txt
```
- Finetuning Training takes about 10 hours in our 1 Tesla V100 GPUs in Googlecolab.
- If you experience out-of-memory errors, you can reduce the batch size.
- You can view progress on GoogleColab by the sidebar.
- After training, you can test checkpoints by each step.


- Select best model for hyperparametric Beamsearch.
```
!python generate.py --model_path email/checkpoint1000/   --search top_kp --content_text "the Chinese text you want to input"

```

### Training Slogan model
- [Setup](#environment-setup) your environment
- In the experiment file, train with the pretrained  model
```
cd /content/drive/'My Drive'/'sinovation'/'project_gpt2_demo'/

from transformers import Trainer, TrainingArguments, AutoModelWithLMHead 
from modeling_gpt2 import GPT2Config,GPT2LMHeadModel

config = GPT2Config.from_pretrained("/content/drive/My Drive/sinovation/project_gpt2_demo/model")
model = GPT2LMHeadModel.from_pretrained("/content/drive/My Drive/sinovation/project_gpt2_demo/model", config=config)
model.num_parameters()
```
- Finetuning Training takes about 5 hours in our 1 Tesla V100 GPUs in Googlecolab.
- If you experience out-of-memory errors, you can reduce the batch size.
- You can view progress on GoogleColab by the sidebar.
- After training, you can test checkpoints by each step.
- Sogan dataset contain around 3,000 wellknown slogans from TV programs and I did manully check and filtering.


### Training with Bert based model (*unofficial*)
- [Setup](#environment-setup) your environment
- From the experiment directory, run

```
!pip install transformers #==3.1.0
!pip install boto3
!pip install urllib3
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertTokenizer, BertModel, AutoModel, AlbertForMaskedLM

import torch 

tokenizer = BertTokenizer.from_pretrained("clue/roberta_chinese_large")
model = BertModel.from_pretrained("clue/roberta_chinese_large")
```
- We did mask toking experiment with Bert based model comparing to the GPT based model.
- It is well suit to generate the topical related text comparing to GPT model, but has stronger requirment about the amount of the input mask tokens.
- 1-3 Mask tokens is well suited.

```some generated results
outputstring=Masktoken_bert("我x想去国家大剧xxx。希x你也有时x一起过x。")
分词ID: tensor([[ 101, 2769,  103, 2682, 1343, 1744, 2157, 1920, 1196,  103,  103,  103,
          511,  102, 2361,  103,  872,  738, 3300, 3198,  103,  671, 6629, 6814,
          103,  511,  102]])
generated results:
我也想去国家大剧院看看。希望你也有时间一起过来。
```

```
outputstring=Masktoken_bert("我x想x去x国x家x大x剧xxx。希x你也有时x一起过x。")
分词ID: tensor([[ 101, 2769,  103, 2682,  103, 1343,  103, 1744,  103, 2157,  103, 1920,
          103, 1196,  103,  103,  103,  511,  102, 2361,  103,  872,  738, 3300,
         3198,  103,  671, 6629, 6814,  103,  511,  102]])
generated results:
我很想要去美国一家的大的剧院看看。希望你也有时间一起过来。
```

## License
Licensed under an MIT license.



