# Natural Language Processing and Categorizing Arabic articles

## Introduction:
The goal of this project is to use Natural Language Processing and Unsupervised Learning models to categorize them into sections. After refining the approach. In this project, we will analyze the articles and categorize them into groups such as political and geographic articles.

## Data description:
dataset is provided by kaggle website, dataset contains titles and introductions that are arabic text and each artical has numeric unique index. 

## Size of Data:
data provided by kaggle <br>
- 	Max words number is 1972 words <br>
-	Min words number is 116 words<br>
-	Max token length is 397 tokens<br>
-	Min token length is 19 tokens <br>
-	Number of 1050 rows <br>
-	Number of columns 3 <br>
     -	title<br>
     -	Introduction <br>
     -	index<br>

### Exploratory data analysis (EDA)
-	Remove NULLS and Duplicate values
-	Merge Title and Introduction columns
-	Remove Unnecessary columns
-	Add Artical column

 
### Text preprocessing: <br>
-	Remove Non-Arabic Char<br>
-	Remove extra spaces<br>
-	Remove Arabic Diacritics<br>
-	Remove Arabic Numbers <br>
-	Normalize Arabic Letters <br>ك → گ
-	Remove Arabic Punctuations ؟ ، . /  !<br>
-	Remove Arabic Stop Word (350word)<br>
-	Remove Repeating Char <br>وو ---> و , يي ---> ي 

## Algorithm: 

###  Embedding
-	CountVectorizer<br>
-	TF-IDF<br>
### 	Topic Modelling
-	LatentDirichletAllocation(LDA)<br>
-	LatentSemanticAnalysis(LSA)<br>
-	Non-NegativeMatrixFactorization(NMF)<br>
-	CorEx<br>
-	Clustring and (Elbow curve and TSNE) <br>


Best model is NMF with Count Vectorizer because it generates manful topics, but other models give unrelated terms for each topic <br>
- Most common words in topic 1<br>
 عام, عمل, عالم, شخص, حيا, تم, تاريخ, فتر, بدء, عمر, يوم, خاص
- Most common words in topic 2<br>
 مدين, دول, تقع, جنوب, تعتبر, مساح, بحر, شرق, عرب, جزير, سكان, غرب
- Most common words in topic 3<br>
 جسم, غذاء, تناول, انس, الصح, طبيع, انواع, حرار, يعتبر, لذل, مواد, لان 
- Most common words in topic 4<br>
 اصاب, حال, علاج, اعراض, مرض, مصاب, شخص, عدوي, فيروس, جهاز, الدم, تظهر
- Most common words in topic 5<br>
طريق, ماء, يتم, استخدام, بشر, شعر, لمد, مكون, زيت, وصف, غسل, حصول

 
### We can notice that topics are:
-	Topic 0: تاريخ
-	Topic 1: جغرافيا
- Topic 2: الغذاء والطعام
- Topic 3: الطب
-	Topic 4: الصحة والجمال


## Recommendation system
### •	Document Similarity with Count vectorizer 
Below, the article that we will predict similar articles to:<br>

صيد الأسماك في مصر
مصر من البلدان العربية الغنية بمصادرها السمكية سواء التي تعيش في المياه العذبة أو بالمياه المالحة، وقد تم إنشاء الكثير من المشاريع السمكية بسبب توفر المياه على طول مجرى نهر النيل وقناة السويس، حيث تتم تربية أنواع محددة من الأسماك بقصد تصديرها إلى الخارج.


-	Similar articles:<br>
  -	First article:<br>
أكبر مدينة في العراق
تعد بغداد عاصمة دولة العراق و أكبر مدينة فيها و ثاني أكبر مدينة على مستوى الوطن العربي و غرب قارة اسيا ، حيث يبلغ عدد سكانها حوالي 7,180,889 نسمة ، تقع بغداد على طول نهر دجلة ، و بسبب موقعها المميز ازدهرت و تطورت المدينة بشكل سريع ، كما ساعدها موقعها من السيطرة على طرق التجارة الواقعة على طول نهر دجلة ، و يقسم نهر دجلة المدينة الى قسمين هما : الرصافة الواقعة في الجهة الشرقية ، و الكرخ الواقعة في الجزء الغربي ، و تعتبر مدينة بغداد مركزا للثقافة العربية ، و تضم المدينة مجموعة من المعالم التاريخية و الأثرية السياحية أهمها : نصب الشهيد ، حديقة الحيوانات ، المتحف الوطني .

  -	Second article:<br>
الرياض عاصمة السعودية
تعد مدينة الرياض (بالإنجليزية : Riyadh ) العاصمة الرسمية للمملكة العربية السعودية و أكبر مدينة فيها ، و تعتبر من المدن العربية الكبرى من حيث المساحة الجغرافية . تقع في المنطقة الوسطى بالقرب من المنطقة الشرقية على هضبة في القسم الشرقي من هضبة نجد في منتصف شبه الجزيرة العربية، و يصل ارتفاعها فوق سطح البحر إلى 600م ، و تبعد الرياض عن مدينة جازان مسافة 1245كم من الجهة الشمالية الشرقية ، و عن مدينة الدمام بحوالي 389كم ، و تبعد عن مكة المكرمة مسافة 880كم من الجهة الشرقية ، و الآتي معلومات عن مدينة الرياض :


  -	Third article:<br>
موقع دولة العراق

تقع العراق في الجزء الجنوبي الغربي من قارة اسيا، وتشترك العراق في حدودها الشمالية مع دولة تركيا ، و يحدها من الجهة الشرقية دولة إيران ، و من الجهة الجنوبية الكويت و المملكة العربية السعودية ، و سوريا و الأردن و المكلة العربية السعودية من الجهة الغربية ، و تبلغ مساحة البلاد حوالي 438,317 كيلومترا مربعا ، و يصل عدد سكانها الى ما يقارب 32,585,000 نسمة حسب إحصاءات 2014م ، و تعتبر مدينة بغداد عاصمتها الرسمية ، و يعد مناخها جافا صحراويا في أغلب مناطق الدولة .



## Tools
### 	Software Platform:<br>
-	JupyterNotebook<br>
### 	Programming Language:<br>
-	Python<br>
### 	Python Libraries:<br>
- Sklearn<br>
- Nltk<br>
-	pyarabic.araby<br>
### 	Data manipulation libraries<br>
-	Pandas<br>
-	Numpy<br>
### 	Visualization<br>
-	Matplotlib<br>
-	Seaborn<br>
-	Canva<br>


## Future Work
-	More Recommendation systems<br>
-	Optimizing Arabic Python Libraries<br>
-	Remove More Arabic Stop Word<br>
