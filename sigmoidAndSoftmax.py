import re,operator,numpy as np
from numpy import exp, array, random, dot
import matplotlib.pyplot as plt
def sigmoid(x):
  return 1/(1+exp(-x*1.0))
def sigmoid_derivative(x):
  return x*(1-x)
def tanh(x):
  return((exp(x*1.0)-exp(-x*1.0))/(exp(x*1.0)+exp(-x*1.0)))
def tanh_derivative(x):
  return(1-x**2)
def softmax(X):
  exps = np.exp(X)
  return exps / np.sum(exps)
def NeuralNetwork_Sigmoid(X,Y,X_test,Y_test,weights1,weights2,weights3):
  testnumber = len(X)
  checknumber = len(X_test)
  insampleerror = 5574
  cnt = 0
  a=[]
  b=[]
  itr=[]
  while(cnt<=4500):
    for aditya in range(testnumber):
      if(cnt>4500):
        break;
      itr.append(cnt)
      cnt+=1
      input1 = X[aditya]
      layer2 = np.array(sigmoid(dot(input1, weights1)))
      layer3 = np.array(sigmoid(dot(layer2, weights2)))
      output = np.array(softmax(dot(layer3, weights3)))
      output2 = np.array(dot(layer3,weights3))
      exps = exp(output2)
      p = (np.sum((exps)))**2
      p = exps[0]*exps[1]/p
      del4=[(Y[aditya][0]-output[0])*p,(Y[aditya][1]-output[1])*p]
      del4 = np.array(del4)
      del3 = np.array(dot(weights3, del4.T)*(sigmoid_derivative(layer3).T))
      del2 = np.array(dot(weights2, del3)*(sigmoid_derivative(layer2).T))
      layer3=np.array([layer3]).reshape(1,50)
      del4=np.array([del4]).reshape(1,2)
      layer2=np.array([layer2]).reshape(1,100)
      del3=np.array([del3]).reshape(50,1)
      del2=np.array([del2]).reshape(100,1)
      input1=np.array([input1]).reshape(1,2000)  
      weights1 += dot(input1.T, del2.T)*.01 
      weights2 += dot(layer2.T, del3.T)*.01
      weights3 += dot(layer3.T, del4)*.01
      layer2 = sigmoid(dot(X, weights1))
      layer3 = sigmoid(dot(layer2, weights2))
      output = softmax(dot(layer3, weights3))
      out = []
      for i in range(len(output)):
         if((output[i][0])<(output[i][1])):
            out.append(Y[i][1])
         else:
            out.append(Y[i][0])
      lol = 0
      for i in range(len(Y)):
         lol+=abs(out[i]-Y[i][0])**2
  
      a.append(lol)
      layer2 = sigmoid(dot(X_test, weights1))
      layer3 = sigmoid(dot(layer2, weights2))
      output = softmax(dot(layer3, weights3))
      out = []
      for i in range(len(output)):
         if(exp(output[i][0])<exp(output[i][1])):
          out.append(Y_test[i][1])
         else:
          out.append(Y_test[i][0])
      lol = 0
      for i in range(len(Y_test)):
         lol+=abs(out[i]-Y_test[i][0])**2
      b.append(lol)
  insampleerror = min(a)
  outsampleerror = min(b)
  optimal = b.index(outsampleerror)
  print("***********************************************************")
  print("In Sample Error")
  accuracy = 100-(insampleerror/len(X))*100
  print("Percent Accuracy =",accuracy,"%")
  print("Number of msg incorrectly predicted",insampleerror)
  print("Number of msg correctly predicted",len(X)-insampleerror)
  print("***********************************************************")
  print()
  print("***********************************************************")
  print("***********************************************************")
  print("Out Sample Error")
  accuracy = 100-(outsampleerror/len(X_test))*100
  print("Percent Accuracy =",accuracy,"%")
  print("Number of msg incorrectly predicted",outsampleerror)
  print("Number of msg correctly predicted",len(X_test)-outsampleerror)
  print("Optimal Number of Iteration at which outsample error is minimum =",optimal)
  print("***********************************************************")
  print()
  plt.plot(itr,a)
  plt.xlabel("Iterations")
  plt.ylabel("Number of Incorrect predictions")
  plt.title("In Sample Error vs Iterations(Softmax)")
  plt.show()
  plt.plot(itr,b)
  plt.xlabel("Iterations")
  plt.ylabel("Number of Incorrect predictions")
  plt.title("Out Sample Error vs Iterations(Softmax)")
  plt.show()
  plt.plot(itr,a,'r',label="Insample Error")
  plt.plot(itr,b,'b',label="Outsample Error")
  plt.title("Insample && Outsample vs Iterations(Softmax)")
  plt.xlabel("Iterations")
  plt.ylabel("Error")
  plt.show()
      
regexp = re.compile(r"[^aeiouy]*[aeiouy]+[^aeiouy](\w*)")
def func1(word):
    if word.startswith('gener') or word.startswith('arsen'):
        return 5
    if word.startswith('commun'):
        return 6
    match = regexp.match(word)
    if match:
        return match.start(1)
    return len(word)
def func2(word):
    match = regexp.match(word, func1(word))
    if match:
        return match.start(1)
    return len(word)
def func3(word):
    if len(word) == 2:
        if re.match(r"^[aeiouy][^aeiouy]$", word):
            return True
    if re.match(r".*[^aeiouy][aeiouy][^aeiouywxY]$", word):
        return True
    return False
def func4(word):
    if func3(word):
        if func1(word) == len(word):
            return True
    return False
def func5(word):
    if word.endswith("'s'"):
        return word[:-3]
    if word.endswith("'s"):
        return word[:-2]
    if word.endswith("'"):
        return word[:-1]
    return word
def func6(word):
    if word.endswith('sses'):
        return word[:-4] + 'ss'
    if word.endswith('ied') or word.endswith('ies'):
        if len(word) > 4:
            return word[:-3] + 'i'
        else:
            return word[:-3] + 'ie'
    if word.endswith('us') or word.endswith('ss'):
        return word
    if word.endswith('s'):
        preceding = word[:-1]
        if re.search(r"[aeiouy].", preceding):
            return preceding
        return word
    return word
def func7(word, r1):
    if word.endswith('eedly'):
        if len(word) - 5 >= r1:
            return word[:-3]
        return word
    if word.endswith('eed'):
        if len(word) - 3 >= r1:
            return word[:-1]
        return word
    def func8(word):
        doubles = ['bb', 'dd', 'ff', 'gg', 'mm', 'nn', 'pp', 'rr', 'tt']
        for double in doubles:
            if word.endswith(double):
                return True
        return False
    def func7_double_check(word):
        if word.endswith('at') or word.endswith('bl') or word.endswith('iz'):
            return word + 'e'
        if func8(word):
            return word[:-1]
        if func4(word):
            return word + 'e'
        return word
    suffixes = ['ed', 'edly', 'ing', 'ingly']
    for suffix in suffixes:
        if word.endswith(suffix):
            preceding = word[:-len(suffix)]
            if re.search(r"[aeiouy]", preceding):
                return func7_double_check(preceding)
            return word
    return word
def func9(word):
    if word.endswith('y') or word.endswith('Y'):
        if word[-2] not in 'aeiouy':
            if len(word) > 2:
                return word[:-1] + 'i'
    return word
def func10(word, r1):
    def func10_double_check(end, repl, prev):
        if word.endswith(end):
            if len(word) - len(end) >= r1:
                if prev == []:
                    return word[:-len(end)] + repl
                for p in prev:
                    if word[:-len(end)].endswith(p):
                        return word[:-len(end)] + repl
            return word
        return None
    triples = [('ization', 'ize', []),('ational', 'ate', []),('fulness', 'ful', []),('ousness', 'ous', []),('iveness', 'ive', []),('tional', 'tion', []),('biliti', 'ble', []),('lessli', 'less', []),('entli', 'ent', []),('ation', 'ate', []),('alism', 'al', []),('aliti', 'al', []),('ousli', 'ous', []),('iviti', 'ive', []),('fulli', 'ful', []),('enci', 'ence', []),('anci', 'ance', []),('abli', 'able', []),('izer', 'ize', []),('ator', 'ate', []),('alli', 'al', []),('bli', 'ble', []),('ogi', 'og', ['l']),('li', '', ['c', 'd', 'e', 'g', 'h', 'k', 'm', 'n', 'r', 't'])]
    for trip in triples:
        attempt = func10_double_check(trip[0], trip[1], trip[2])
        if attempt:
            return attempt
    return word
def func12(word, r1, r2):
    def func12_double_check(end, repl, r2_necessary):
        if word.endswith(end):
            if len(word) - len(end) >= r1:
                if not r2_necessary:
                    return word[:-len(end)] + repl
                else:
                    if len(word) - len(end) >= r2:
                        return word[:-len(end)] + repl
            return word
        return None
    triples = [('ational', 'ate', False),('tional', 'tion', False),('alize', 'al', False),('icate', 'ic', False),('iciti', 'ic', False),('ative', '', True),('ical', 'ic', False),('ness', '', False),('ful', '', False)]
    for trip in triples:
        attempt = func12_double_check(trip[0], trip[1], trip[2])
        if attempt:
            return attempt
    return word
def func13(word, r2):
    delete_list = ['al', 'ance', 'ence', 'er', 'ic', 'able', 'ible', 'ant', 'ement', 'ment', 'ent', 'ism', 'ate', 'iti', 'ous', 'ive', 'ize']
    for end in delete_list:
        if word.endswith(end):
            if len(word) - len(end) >= r2:
                return word[:-len(end)]
            return word
    if word.endswith('sion') or word.endswith('tion'):
        if len(word) - 3 >= r2:
            return word[:-3]
    return word
def func14(word, r1, r2):
    if word.endswith('l'):
        if len(word) - 1 >= r2 and word[-2] == 'l':
            return word[:-1]
        return word
    if word.endswith('e'):
        if len(word) - 1 >= r2:
            return word[:-1]
        if len(word) - 1 >= r1 and not func3(word[:-1]):
            return word[:-1]
    return word
def stemWord(word):
  if len(word) <= 2:
    return word
  if word.startswith("'"):
    word = word[1:]
  exceptional_forms = {'skis': 'ski','skies': 'sky','dying': 'die','lying': 'lie','tying': 'tie','idly': 'idl','gently': 'gentl','ugly': 'ugli','early': 'earli','only': 'onli','singly': 'singl','sky': 'sky','news': 'news','howe': 'howe','atlas': 'atlas','cosmos': 'cosmos','bias': 'bias','andes': 'andes'}
  if word in exceptional_forms:
    return exceptional_forms[word]
  if word.startswith('y'):
    word = 'Y' + word[1:]
  word =re.sub(r"([aeiouy])y", '\g<1>Y', word)
  r1 = func1(word)
  r2 = func2(word)
  word = func5(word)
  word = func6(word)
  exceptional_early_exit_post_1a = ['inning', 'outing', 'canning', 'herring', 'earring', 'proceed', 'exceed', 'succeed']
  if word in exceptional_early_exit_post_1a:
    return word
  word = func7(word, r1)
  word = func9(word)
  word = func10(word, r1)
  word = func12(word, r1, r2)
  word = func13(word, r2)
  word = func14(word, r1, r2)
  word.replace('Y', 'y')
  return word    


typ=[]
msg=[]
s = list(map(lambda x:x.strip('\n'),open('spam_classifier_data.txt','r').readlines()))
for i in range(5574):
  if(s[i][0]=='h'):
    typ.append("ham")
    for j in range(3,len(s[i])):
      if(s[i][j]!="\t"):
        break
    temp = s[i][j:]
    msg.append(temp)
  else:
    typ.append("spam")
    for j in range(4,len(s[i])):
      if(s[i][j]!="\t"):
        break
    temp = s[i][j:]
    msg.append(temp)
for i in range(5574):
  msg[i] = re.sub('[^ a-z\'\"]', '', msg[i])
  msg[i] = msg[i].strip().split(' ')
  t = len(msg[i])
  for j in range(t):
    msg[i][j] = msg[i][j].lower()

stopwords = ["a","Ok","ok", "about", "above","wen","wat","u","ur","me","once","saturday","sunday","wednesday","yo","above", "across" ,"after","wat", "Go","afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]
for i in range(5574):
  t = len(msg[i])
  temp=[]
  for j in range(t):
    if(msg[i][j] in stopwords):
      temp.append(msg[i][j])
  for j in temp:
    msg[i].remove(j)
s=[]
for i in range(5574):
  t=len(msg[i])
  for j in range(t):
    msg[i][j]=stemWord(msg[i][j])
    s.append(msg[i][j])
t=set(s)
di={}
for i in t:
  di[i] = s.count(i)
di = sorted(di.items(), key=operator.itemgetter(1))
finalwords = []
cnt = 0
l = len(di)
for key in di:
  finalwords.append(key[0])
finalwords.reverse()
finalwords = finalwords[:2000]
di={}
for i in range(2000):
  di[finalwords[i]] = i
testnumber = int(.8*5574)
X=[]
Y=[]
X_test=[]
Y_test=[]
for i in range(testnumber):
  t = len(msg[i])
  temp = [0 for kk in range(2000)]
  for j in range(t):
    ch = msg[i][j]
    try:
      p = di[ch]
      temp[p]+=1
    except KeyError:
      p = 0
  X.append(temp)
  if(typ[i][0]=='h'):
    pp = [1,0]
  else:
    pp = [0,1]
  Y.append(pp)
for i in range(testnumber,5574):
  t = len(msg[i])
  temp = [0 for kk in range(2000)]
  for j in range(t):
    ch = msg[i][j]
    try:
      p = di[ch]
      temp[p]+=1
    except KeyError:
      p = 0
  X_test.append(temp)
  if(typ[i][0]=='h'):
    pp = [1,0]
  else:
    pp = [0,0]

  Y_test.append(pp)
X = np.array(X)
Y = np.array(Y)
random.seed(1)
X_test = np.array(X_test)
Y_test = np.array(Y_test)
l2 = 100
l3 = 50
weights1 = np.array(2 * random.random((2000, l2)) -1)
weights2 = np.array(2 * random.random((l2, l3)) -1)
weights3 = np.array(2 * random.random((l3, 2)) -1)
print("USING SOFTMAX FUNCTION")
print("***********************************************************")
NeuralNetwork_Sigmoid(X,Y,X_test,Y_test,weights1,weights2,weights3)
print("***********************************************************")
print()
print()
print("***********************************************************")

