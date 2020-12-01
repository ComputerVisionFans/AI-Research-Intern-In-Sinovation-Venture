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


