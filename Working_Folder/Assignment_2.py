def naiveSoftmaxLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset
):
    """ Naive Softmax loss & gradient function for word2vec models

    Implement the naive softmax loss and gradients between a center word's 
    embedding and an outside word's embedding. This will be the building block
    for our word2vec models.

    Arguments:
    centerWordVec -- numpy ndarray, center word's embedding
                    (v_c in the pdf handout)
    outsideWordIdx -- integer, the index of the outside word
                    (o of u_o in the pdf handout)
    outsideVectors -- outside vectors (rows of matrix) for all words in vocab
                      (U in the pdf handout)
    dataset -- needed for negative sampling, unused here.

    Return:
    loss -- naive softmax loss
    gradCenterVec -- the gradient with respect to the center word vector
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    (dJ / dU)
    """

    ### YOUR CODE HERE
    
    # Loss function
    
    x = np.dot(outsideVectors,centerWordVec)
    y_hat =  softmax(x)
    loss = -np.log(y_hat[outsideWordIdx])
    
    # Gradient of vc = Ut (y-yhat)
    y = np.zeros(y_hat.shape)
    y[outsideWordIdx] = 1
    Ind_grad  = y - y_hat
    gradCenterVec = np.dot(Ind_grad,outsideVectors)
    
    # Gradient of Outisde Matrix U =  (yhat - y)T.Vc
    v_exp = (np.expand_dims(centerWordVec, axis=0))
    Ind_grad_exp = (np.expand_dims((y_hat-y), axis=0))
    gradOutsideVecs = np.dot(np.transpose(Ind_grad_exp),v_exp)
    
    ### Please use the provided softmax function (imported earlier in this file)
    ### This numerically stable implementation helps you avoid issues pertaining
    ### to integer overflow. 


    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs
    
    
    def negSamplingLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset,
    K=10
):
    """ Negative sampling loss function for word2vec models

    Implement the negative sampling loss and gradients for a centerWordVec
    and a outsideWordIdx word vector as a building block for word2vec
    models. K is the number of negative samples to take.

    Note: The same word may be negatively sampled multiple times. For
    example if an outside word is sampled twice, you shall have to
    double count the gradient with respect to this word. Thrice if
    it was sampled three times, and so forth.

    Arguments/Return Specifications: same as naiveSoftmaxLossAndGradient
    """

    # Negative sampling of words is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    ### YOUR CODE HERE
    ### Please use your implementation of sigmoid in here.
    
    # Loss function
    
    negSampleWordIndices = getNegativeSamples(outsideWordIdx, dataset, K)
    indices = [outsideWordIdx] + negSampleWordIndices
    x = np.dot(outsideVectors[outsideWordIdx],centerWordVec)
    correct_part = -np.log(sigmoid(x))
    y = np.dot(outsideVectors[negSampleWordIndices],centerWordVec)
    Incorrect_part = np.sum(np.log(sigmoid(-y)))
    loss = correct_part + Incorrect_part
       
    ## GradCenterVec
    ## −(1 − x)u0 +sum((1 − y)uk)

    gradCenterVec = -np.dot((1-x),outsideVectors[outsideWordIdx]) + np.sum(np.dot((1-x),outsideVectors[negSampleWordIndices]))
    
    ## gradOutsideVecs
    ## u0 = −(1 − x)vc , uw =  (1 − y)vc
    
    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs
    
    
    
    def sigmoid(x):

    if x >= 0:
        z = exp(-x)
        return 1 / (1 + z)
    else:
        z = exp(x)
        return z / (1 + z)

    return s
