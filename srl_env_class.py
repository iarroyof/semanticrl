

import gym
from gym import spaces



class weigher(object):
    def __init__(self, ret_np=False):
        self.ret_np = ret_np
        self.model = TfidfVectorizer()
        
    def fit(self, input_text):
        if isinstance(input_text, str):
            with open(input_text) as f:
                self.model.fit(f)
        else:
            self.model.fit(input_text)
            
        self.vocab = self.model.vocabulary_
        self.prepr = self.model.build_preprocessor()
        self.toker = self.model.build_tokenizer()
        
    def tokenize(self, string):
        return self.toker(self.prepr(string))
    
    def tfidf(self, St):
        
        sparse_wv = self.model.transform([St])
        st = []
        for w in self.tokenize(St):
            try:
                widx = self.vocab[w]
                st.append(sparse_wv[0, widx])
            except KeyError:
                st.append(0.0)
        
        return np.array(st) if self.ret_np else st


class textEnv(gym.Env):
    """Custom text environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, input_file_name, wsize=7, beta_rwd=1.5, gamma = 0.8, 
                                        sample_size=20, traject_length=100, 
                                        reward_smooth=True, n_trajects=10):
        super(textEnv, self).__init__()

        self.weiger = weigher(ret_np=True)
        self.weiger.fit(input_file_name)
        self.total_bytes = os.stat(input_file_name).st_size 
        self.file = open(input_file_name)
        assert wsize > 3  # No context size smaller than 3 allowed
        self.w_size = wsize
        self.tlegth = traject_length
        self.tcount = 0
        self.reward_smooth = True
        self.gamma = gamma
        self.beta = beta_rwd
        self.rand_byte = random.randint(0, self.total_bytes)
        self.current_step = 0
        self.sample_size = sample_size
        self.n_trajects = n_trajects
        
        try:
            self.n_gram_model = TfidfVectorizer(analyzer='char_wb', 
                                                ngram_range=(1,3))
        except TypeError:
            self.n_gram_model = TfidfVectorizer(analyzer='char', 
                                                ngram_range=(1,3))
        token_pattern = re.compile('\\w+')
        self.tokenize = lambda s: token_pattern.findall(s)
        self.char_prep = self.n_gram_model.build_preprocessor()
        self.char_analyzer = self.n_gram_model.build_analyzer()
        
    def char_tokenizer(self, s):
        """Gives character n-gram tokens.
        args: s: string (a context window in string form).
        rets: a list of strings, each being an n-gram.
        """
        return [ng for ng in self.char_analyzer(" ".join(self.tokenize(
                    self.char_prep(s)))) 
                        if not ng in ['', ' ']]
        
    def reset(self):
        self.I_XZgY = []
        self.I_XZ = []
        self.tcount = 0
        self.sample_semanticity = []
        self.horizon_semanticity = 0.0
        self.cum_rewards = []
        self.current_byte = random.randint(0, self.total_bytes)
        self.file.seek(self.current_byte)
        self.current_step = 0
        
        return self.next_observation()
    
    def _check_EOF(self):
        self.current_byte = self.file.tell()
        if self.current_byte == self.total_bytes:
            return True
        else:
            return False
        
    def _read_next_context(self):
        #if self._check_EOF():
        #    """If end of file is reached, then go to random line"""
        #    self.file.seek(0)
        #    self.current_byte = random.randint(0, self.total_bytes)
        #    self.file.seek(self.current_byte)
        #    self.file.readline() # skip this line to clear the partial line
        self.lline = []    
        while len(self.lline) < self.w_size:
            """Do not deliver a text line if it is shorter than the allowed 
                window size"""
            if self._check_EOF():
                self.file.seek(0)
                self.current_byte = random.randint(0, self.total_bytes)
                self.file.seek(self.current_byte)
                self.file.readline() # skip this line to clear the partial line
                self._read_next_context()
            else:
                self.lline = self.weiger.tokenize(self.file.readline())
                #self.current_byte = self.file.tell()
                
                """
                Update the current file position, pick up a random context from 
                the current line at sc (start context, measured in tokens), and
                return it as a string."""
        if len(self.lline) - self.w_size > 0:
            self.sc = random.randint(0, len(self.lline) - self.w_size)
        else:
            self.sc = 0

        ctxt = " ".join(self.lline[self.sc:self.sc + self.w_size])
        #print(ctxt)
        return ctxt
                
    def next_observation(self):
    #def _next_state(self):
        """This method reads |D_k| contexts to form a D_k sample in a step, and 
        returns a matrix whose rows are \psi information signals of each context.
        args: no arguments
        rets: tuple(list of string contexts S_t, numpy matrix of \psi signals)
        """
        D_k = []
        #S_k = []
        for _ in range(self.sample_size):
            context = self._read_next_context()
            D_k.append((context, self.weiger.tfidf(context)))
            #S_k.append(self.weiger.tfidf(context))
        
        return D_k  #, S_k
    
    def conditioned_MI(self, X, Y, Z):
        """Compute conditioned mutual information with respect to the hypothesis
        of that Y = y is the head for each triplet of the action (sample) step.
        
        args: X: list of strings corresponding to n-grams for the hypothesis of
                    that X = x for each triplet of the current action step.
              Y: list of strings corresponding to n-grams for the hypothesis of
                    that Y = y for each triplet of the current action step.
              Z: list of strings corresponding to n-grams for the hypothesis of
                    that Z = z for each triplet of the current action step.
        rets: float: The method returns the CMI.
        """
        Tn = set(X).intersection(Y).intersection(Z)
        Tu = set(X).union(Y).union(Z)
        XnY = set(X).intersection(Y)
        ZnY = set(Z).intersection(Y)
        
        P_XYZ = len(Tn)/len(Tu)
        P_XZgY = len(Tn)/len(Y)
        P_XgY = len(XnY) / len(Y)
        P_ZgY = len(ZnY) / len(Y)
        
        I_XZgY = P_XYZ * np.log(P_XZgY/(P_XgY * P_ZgY))
        
        return I_XZgY
        
        
    def mutual_info(self, X, Y, Z):
        """Compute mutual information between the hypotheses of that X = x and
        Z = z for each triplet of the action (sample) step. The method needs the
        whole triplets (X, Y, Z) to compute X, Y probabilities within the sample.
        
        args: X: list of strings corresponding to n-grams for the hypothesis of
                    that X = x for each triplet of the current action step.
              Y: list of strings corresponding to n-grams for the hypothesis of
                    that Y = y for each triplet of the current action step.
              Z: list of strings corresponding to n-grams for the hypothesis of
                    that Z = z for each triplet of the current action step.
        rets: float: The method returns the CMI.
        """
        Tu = set(X).union(Y).union(Z)
        XnZ = set(X).intersection(Z)
        
        P_XZ = len(XnZ)/len(Tu)
        P_X = len(X)/len(Tu)
        P_Z = len(Z)/len(Tu)
        
        I_XZ = P_XZ * np.log(P_XZ/(P_X * P_Z))
        
        return I_XZ
        
    def _interpret_action(self, action):
        """Actions 'a' from a sample constitute a step 'action', where 
           args: action: list of dicts [a1, a2,...]\equiv [{Y: list(w1, w2,...), 
                                                            X: list(w1, w2,...), 
                                                         Z: list(w1, w2,...)}, ]
           rets: float: semanticity, and updating of reward domain via 
                        self.I_XZgY and self.I_XZ
        """
        self.Ak = pd.DataFrame(action)
        X = set(sum([x for x in map(self.char_tokenizer, 
                                    map(" ".join, list(self.Ak.X)))], []))
        Y = set(sum([y for y in map(self.char_tokenizer, 
                                    map(" ".join, list(self.Ak.Y)))], []))
        Z = set(sum([z for z in map(self.char_tokenizer, 
                                    map(" ".join, list(self.Ak.Z)))], []))
        I_xz_y = self.conditioned_MI(X, Y, Z)
        I_xz = self.mutual_info(X, Y, Z)
        semanticity = I_xz_y - I_xz
        
        self.I_XZgY.append(I_xz_y)
        self.I_XZ.append(I_xz)
        self.sample_semanticity.append(semanticity)
        
        return semanticity
  

    def _reward_func(self, semanticity):
        
        if not self.reward_smooth:
            return np.heaviside(semanticity, 0.0)
        else:
            return 1.0 / (1.0 + math.exp(-self.beta * semanticity))
        
    def step(self, action=None):
        if action is None:
            self.sample_semanticity.append(0.0)
            obs = self.next_observation()
            reward = 0.0
            done = False
        else:    
            semanticity = self._interpret_action(action)
            self.current_step += 1
            #if self.current_step > self.tlegth:
            #    self.current_step = 0
            #    self.tcount += 1

            gamma_t = (1.0 if self.current_step == 0 
                              else self.gamma ** self.current_step)
            obs = self.next_observation()
            reward = gamma_t * self._reward_func(semanticity)
            #done = self.tcount > self.n_trajects
            done = self.current_step >= self.tlegth
        
        self.cum_rewards.append(reward)
        
        return obs, reward, done, {}
    
    def render(self):
        # Render the environment to the screen
        #self.action_semanticity = self.Iy_xz - self.Ixz  # How meaningful the 
                                                          # structure build by 
                                                          # the agent is.                
        print(f'\nStep: {self.current_step}')
        print(f'Sample Semanticity: {self.sample_semanticity}')
        print(f'Cumulative rewards: {self.cum_rewards}')