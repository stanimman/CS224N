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

    gradCenterVec = -np.dot((1-x),outsideVectors[outsideWordIdx]) + np.sum(np.dot((1-y),outsideVectors[negSampleWordIndices]))
    
    ## gradOutsideVecs
    ## u0 = −(1 − x)vc , uw =  (1 − y)vc
    
    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs

    
    
    
    
   def sigmoid(x):

    s = 1 / (1 + np.exp(-x))
    return s
   

    def skipgram(currentCenterWord, windowSize, outsideWords, word2Ind,
             centerWordVectors, outsideVectors, dataset,
             word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentCenterWord -- a string of the current center word
    windowSize -- integer, context window size
    outsideWords -- list of no more than 2*windowSize strings, the outside words
    word2Ind -- a dictionary that maps words to their indices in
              the word vector list
    centerWordVectors -- center word vectors (as rows) for all words in vocab
                        (V in pdf handout)
    outsideVectors -- outside word vectors (as rows) for all words in vocab
                    (U in pdf handout)
    word2vecLossAndGradient -- the loss and gradient function for
                               a prediction vector given the outsideWordIdx
                               word vectors, could be one of the two
                               loss functions you implemented above.

    Return:
    loss -- the loss function value for the skip-gram model
            (J in the pdf handout)
    gradCenterVecs -- the gradient with respect to the center word vectors
            (dJ / dV in the pdf handout)
    gradOutsideVectors -- the gradient with respect to the outside word vectors
                        (dJ / dU in the pdf handout)
                        naiveSoftmaxLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset
):
 Naive Softmax loss & gradient function for word2vec models

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

    loss = 0.0
    gradCenterVecs = np.zeros(centerWordVectors.shape)
    gradOutsideVectors = np.zeros(outsideVectors.shape)

    ### YOUR CODE HERE
    centerWordVec_Ind = word2Ind[currentCenterWord]
    #print(currentCenterWord)
    #print(centerWordVec_Ind)
    centerWordVec = centerWordVectors[centerWordVec_Ind]
    for i in outsideWords:
        outsideWordIdx = word2Ind[currentCenterWord]
        loss_tmp, gradCenterVec_tmp, gradOutsideVecs_tmp = word2vecLossAndGradient(centerWordVec,outsideWordIdx,outsideVectors,dataset)
        loss = loss+loss_tmp
        gradCenterVecs = gradCenterVecs + gradCenterVec_tmp
        gradOutsideVectors = gradOutsideVectors + gradOutsideVecs_tmp
    #print(centerWordVec)
    ### END YOUR CODE

    return loss, gradCenterVecs, gradOutsideVectors
