# ML project template

This template can be used as a starting point for any ML project. It's based on Abhishek Thakur's book Approaching (Almos) Any Machine Learning Problem, which is an incredible resource, not only for this template, but also for a lot of great tips and tricks of how to handle ML projects in general. You can find a link to the github repo [here](https://github.com/abhishekkrthakur/approachingalmost). I highly recommend checking it out.

## What's different about this repo?

Well, let's first understand what this repo is helpful for. Once you get familiar with working with Machine Learning, you will notice that you will tend to repeat a few processes a lot. For example, once you are done with preprocessing your data, you will need to start training models and performing cross validation to tune their hyperparameters.

That process tends to be very repetitive with you going back and forth and changing the models around and the hyperparameters as well. In his book, Abishek shows a very simple process for training an ML model using the terminal and a few extra commands. I've found this really useful and through the years have found some extra additions to him process that have served me well.

That's basically what is different from what he proposes. I have added a model dictionary function that you can modify to alter the parameters of the your ML model. The function then passes the model_params dict as arguments to the actual ML model. I have added a few tips in comment form on how to execute these scripts as well.

## Next steps:

- I'm planning to add the possibility to train a model on all the dataset's fold on a different bash script that you can run from the terminal, similar to what Abishek does on his book.

- I'm also planning to add a pipeline for easily saving the parameters and results to a text file so that you can also track your models, so be on the lookout for that.

Let me know if you are interested in anythiung else!
