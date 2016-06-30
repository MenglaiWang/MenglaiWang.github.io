---
layout: post
title: Introduction to debugging neural networks
---
Some tips for debugging neural networks

The following advice is targeted at beginners to neural networks, and is based on my experience giving advice to neural net newcomers in industry and at Stanford. Neural nets are fundamentally harder to debug than most programs,because most neural net bugs don't result in type errors or runtime errors. They just cause poor convergence. Especially when you're new, this can be very frustrating! But an experienced neural net trainer will be able to systematically overcome the difficulty in spite of the ubiquitous and seemingly ambiguous error message:

  Performance Error: your neural net did not train well.

To the uninitiated, the message is daunting. But to the experienced, this is a great error. It means the boilerplate coding is out of the way, and it's time to dig in!

### How to deal with NaNs

By far the most common first question I get from students is, "Why am I getting NaNs." Occasionally, this has a complicated answer. But most often, the NaNs come in the first 100 iterations, and the answer is simple: your learning rate is too high. When the learning rate is very high, you will get NaNs in the first 100 iterations of training. Try reducing the learning rate by a factor of 3 until you no longer get NaNs in the first 100 iterations. As soon as this works, you'll have a pretty good learning rate to get started with. In my experience, the best heavily validated learning rates are 1-10x below the range where you get NaNs.

If you are getting NaNs beyond the first 100 iterations, there are 2 further common causes.
1) If you are using RNNs, make sure that you are using "gradient clipping", which caps the global L2 norm of the gradients. RNNs tend to produce gradients early in training where 10% or fewer of the batches have learning spikes, where the gradient magnitude is very high. Without clipping, these spikes can cause NaNs.
2) If you have written any custom layers yourself, there is a good chance your own custom layer is causing the problems in a division by zero scenario. Another notoriously NaN producing layer is the softmax layer. The softmax computation involves an exp(x) term in both the numerator and denominator, which can divide Inf by Inf and produce NaNs. Make sure you are using a stabilized softmax implementation.

### What to do when your neural net isn't learning anything

Once you stop getting NaNs, you are often rewarded with a neural net that runs smoothly for many thousand iterations, but never reduces the training loss after the initial fidgeting of the first few hundred iterations. When you're first constructing your code base, waiting for more than 2000 iterations is rarely the answer. This is not because all networks can start learning in under 2000 iterations. Rather, the chance you've introduced a bug when coding up a network from scratch is so high that you'll want to go into a special early debugging mode before waiting on high iteration counts. The name of the game here is to reduce the scope of the problem over and over again until you have a network that trains in less than 2000 iterations. Fortunately, there are always 2 good dimensions to reduce complexity.

1) Reduce the size of the training set to 10 instances. Working neural nets can usually overfit to 10 instances within just a few hundred iterations. Many coding bugs will prevent this from happening. If you're network is not able to overfit to 10 instances of the training set, make sure your data and labels are hooked up correctly. Try reducing the batch size to 1 to check for batch computation errors. Add print statements throughout the code to make sure things look like you expect. Usually, you'll be able to find these bugs through sheer brute force. Once you can train on 10 instances, try training on 100. If this works okay, but not great, you're ready for the next step.

2) Solve the simplest version of the problem that you're interested in. If you're translating sentences, try to build a language model for the target language first. Once that works, try to predict the first word of the translation given only the first 3 words of the source. If you're trying to detect objects in images, try classifying the number of objects in each image before training a regression network. There is a trade-off between getting a good sub-problem you're sure the network can solve, and spending the least amount of time plumbing the code to hook up the appropriate data.Creativity will help here.

The trick to scaling up a neural net for a new idea is to slowly relax the simplifications made in the above two steps. This is a form of coordinate ascent, and it works great. First, you show that the neural net can at least memorize a few examples. Then you show that it's able to really generalize to the validation set on a dumbed down version of the problem. You slowly up the difficulty while making steady progress. It's not as fun as hotshotting it the first time Karpathy style, but at least it works. At some point, you'll find the problem is difficult enough that it can no longer be learned in 2000 iterations. That's great! But it should rarely take more than 10 times the iterations of the previous complexity level of the problem. If you're finding that to be the case, try to search for an intermediate level of complexity.

### Tuning hyperparameters

Now that your networks is learning things, you're probably in pretty good shape. But you may find that your network is just not capable of solving the most difficult versions of your problem. Hyperparameter tuning will be key here. Some people who just download a CNN package and ran it on their dataset will tell you hyperparameter tuning didn't make a difference. Realize that they're solving an existing problem with an existing architecture. If you're solving a new problem that demands a new architecture, hyperparameter tuning to get within the ballpark of a good setting is a must. You're best bet is to read a hyperparameter tutorial for your specific problem, but I'll list
a few basic ideas here for completeness.

* Visualization is key. Don't be afraid to take the time to write yourself nice visualization tools throughout training. If your method of visualization is watching the loss bump around from the terminal,
consider an upgrade.

* Weight initializations are important. Generally, larger magnitude initial weights are a good idea, but too large will get you NaNs. Thus, weight initialization will need to be simultaneously tuned with the learning rate.

* Make sure the weights look "healthy". To learn what this means, I recommend opening weights from existing networks in an ipython notebook. Take some time to get used to what weight histograms should look like for your components in mature nets trained on standard datasets like ImageNet or the Penn Tree Bank.

* Neural nets are not scale invariant w.r.t. inputs, especially when trained with SGD rather than second order methods, as SGD is not a scale-invariant method.  Take the time to scale your input data and output labels in the same way that others before you have scaled them.

* Decreasing your learning rate towards the end of training will almost always give you a boost. The best decay schedules usually take the form: after k epochs, divide the learning rate by 1.5 every n epochs,
where k > n.

* Use hyperparameter config files, although it's okay to put hyperparameters in the code until you start trying out different values. I use json files that I load in with a command line argument as in
https://github.com/Russell91/tensorbox, but the exact format is not important. Avoid the urge to refactor your code as it becomes a hyperparameter loading mess! Refactors introduce bugs that cost you
training cycles, and can be avoided until after you have a network you like.

* Randomize your hyperparameter search if you can afford it. Random search generates hyperparmeter combinations you wouldn't have thought of and removes a great deal of effort once your intuition is already
trained on how to think about the impact of a given hyperparameter.

### Conclusion

Debugging neural nets can be more laborious than traditional programs because almost all errors get projected onto the single dimension of overall network performance. Nonetheless, binary search is still your friend. By alternately

1) changing the difficulty of your problem, and

2) using a small number of training examples, you can quickly work through the initial bugs. Hyperparameter tuning and long periods of diligent waiting will get you the rest of the way.
