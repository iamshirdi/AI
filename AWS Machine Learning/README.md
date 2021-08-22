
<img src = "./static/AWS_Machine_Learning_Scholarship_Winner_Badge.PNG" width =900px height=850px  alt ="AWS Scholarship Badge" title ="AWS Scholarship Badge">

# Projects
- AWS Deep Lens - Componst:sorting-classification
- AWS Deep Racer - Reinforcement Learning (SAC and PPO)
- AWS Deep Composer - Generative AI (ARCNN,GAN,Transformers)


# Intro ML
- AI is a technique to simulate human level intelligence
- ML will learn automatically without explicit programming by inspecting data(unsupervised patterns supervised trained data) and reinforcement (maximize reward) 
- In reinforcement learning, the algorithm figures out which actions to take in a situation to maximize a reward (in the form of a number) on the way to reaching a specific goal.
- Model training algorithms work through an interactive process where the current model iteration is analyzed to determine what changes(unsupervised patterns no need complete model in mind) can be made to get closer to the goal. Those changes are made and the iteration continues until the model is evaluated to meet the goals.
- Regression tasks involve predicting some unknown continuous attribute about your data.
- Clustering tasks involve exploring how your data might be grouped together.
- Data inspection for data integrity includes outliers,missing or incomplete, transform data etc
- Impute is a common term referring to different statistical tools which can be used to calculate missing values from your dataset.
- Split data usually for bias-variance trade-off
- Model parameters: Model parameters are settings or configurations the training algorithm can update to change how the model behaves
- Hyperparameters are settings on the model which are not changed during training but can affect how quickly or how reliably the model trains, such as the number of clusters the model should identify.(number of hidden units,learning rate-set before training optimizing weights and bias)
- Loss function measures how far model is from goal
- Classification tasks often use a strongly related logistic model, which adds an additional transformation mapping the output of the linear function to the range [0, 1], interpreted as “probability of being in the target class.
- Tree based model learns to categorize or regress by building an extremely large structure of nested if/else blocks, splitting the world into different regions at each if/else block. Training determines exactly where these splits happen and what value is assigned at each leaf region.
Neural Network model (also called a neural network) is composed of collections of neurons (very simple computational units) connected together by weights (mathematical representations of how much information to allow to flow from one neuron to the next)
- Transformer: A more modern replacement for RNN/LSTMs neural networks, the transformer architecture enables training over larger datasets involving sequences of data.
- RNN/LSTM: Recurrent Neural Networks (RNN) and the related Long Short-Term Memory (LSTM) model types are structured to effectively represent for loops in traditional computing, collecting state while iterating over some object. They can be used for processing sequences of data.

<br>

<img src = "./static/bias vs variance tradeoff.PNG" alt ="Bias vs variance" title ="Underfittinh(bias) vs overfitting(variance)">

<br>
<br>

- Log loss seeks to calculate how uncertain your model is about the predictions it is generating.
- Mean absolute error (MAE): This is measured by taking the average of the absolute difference between the actual values and the predictions. Ideally, this difference is minimal.
- Root mean square error (RMSE): This is similar MAE, but takes a slightly modified approach so values with large error receive a higher penalty. RMSE takes the square root of the average squared difference between the prediction and the actual value.
- Coefficient of determination or R-squared (R^2): This measures how well-observed outcomes are actually predicted by the model, based on the proportion of total variation of outcomes.
- Ridge Regression is a popular type of regularized linear regression that includes an L2 penalty. This has the effect of shrinking the coefficients for those input variables that do not contribute much to the prediction task.
- Techniques exist to modify the data so you can still use linear models in curved graph situations are called kernel methods.
- Microgrenres in book(back of book similar words or book description text for 800 romance books) to find common words (microgenre)-clustering
- words:lowercase,punctations remove- stop words(a,the),data vectorization for bag of words and clustering to find the genre
- evaluation metrics like confusion matrix , silhouete Coefficient(how well data clustered)f1-score etc
- accuracy:You realize the model will see the 'Does not contain spill' class almost all the time, so any model that just predicts “no spill” most of the time will seem pretty accurate.
- precision as answering the question, "Of all predictions of a spill, how many were right?" and recall as answering the question, "Of all actual spills, how many did we detect?"

# AWS ML
<a href = "https://aws.amazon.com/machine-learning/ai-services/?utm_source=Udacity&utm_medium=Webpage&utm_campaign=Udacity%20AWS%20ML%20Foundations%20Course">
<img src = "./static/AWS-ML-Services.PNG" title ="AWS ML Services types">
</a>

- Amazon sagemaker is a PAAS service to build,train and deploy machine learning models

### Computer Vision
- computer vision recognises patterns(edges-rextures-high features-deep learning) to gain understanidng of image like different rules in different positions, conditions etc to detect image
- CV applications-activity recognition, sorting, ocr, content filtering, parking, argument reality: photo apps , segmentation, recognise + where object is
- lambda function spins up the model and performs inference
- Object detection can be used to not only count the number of objects in an image (same and different) but also to identify different classes present in an image
- Activity recognition is used when we need to identify changes which occur over time in a video

### Reinforcement Learning
- Reinfocement learning what actions to take in environment to maximize reward /rech goal (training dog positive/negative)
- Play games, create right balance by making reinforcment agent play again and again -graphs
- upstream turbines slows down it helps capture more energy by downstream turbine using reinforcement learning
- industrial robots, fraud detection, stock trading, autonomous driving
- state is defined the current position within environment that is visible
- In AWS Deep racer image captured is state. and terminal(episode-initial) state is final off-track or obstacle. Feedback is given to agent as numerical feedback
- reward function is numeric assigns scores as reward to different zones in traks
- SAC (only continous state) encourage exploration (data efficent lacks stablilty) unlike Proimity Policy Optimization
- Discrete action space represents all of an agent's possible actions for each state in a finite set of steering angle and throttle value combinations.
- Continuous action space allows the agent to select an action from a range of values that you define for each sta te.
- learning rate is a hyperparameter that controls how many new experiences are counted in learning at each step.A higher learning rate results in faster training but may reduce the model’s quality
- Reward function as a grid mapped over race track in AWS Deep racer
- stero camera left and right gives depth to avoid obstacles
- generative models are unsupervised. Diiscriminating (classify) and generating (generate data learned during training-pattern or distribution etc)

### GAN
- GAN, Auto regressive models and Transformer based models
- GAN putting two neural networks against each other to generate content. Discriminator measures real/fake and gives feedback to generator to generate realistic images
- The training algorithm swaps back and forth between training a generator network (responsible for producing new data) and a discriminator network (responsible for measuring how closely the generator network’s data represents the training dataset).
- Autoregressive models are used to study systems which evolve over time. They rely on previous time series to generate accurate new data. weather prediction
- Transformer is used to study data that is sequential. pixels in image, nlp
- Use the GAN technique to create accompaniment tracks.
Use the AR-CNN technique to modify notes in your input track.
Use the transformers technique to extend your input track by up to 30 seconds.
- The discriminator then takes these generated music tracks and predicts how far they deviate from the real data present in the training dataset. This deviation is called the generator loss. This feedback from the discriminator is used by the generator to incrementally get better at creating realistic output.
- The generator and the discriminator are trained in alternating cycles. The generator learns to produce more and more realistic data while the discriminator iteratively gets better at learning to differentiate real data from the newly created data.
- Generator loss: Measures how far the output data deviates from the real data present in the training dataset.
Discriminator loss: Evaluates how well the discriminator differentiates between real and fake data.
- Autoregressive convolutional neural networks make iterative changes over time to create new data.. AR-CNN used to edit/correct music
- Loss models stabilize helps that no more training can be done
- 
