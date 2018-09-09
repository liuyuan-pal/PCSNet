import random
import numpy as np

class Provider:
    def __init__(self,file_list,batch_size,read_fn,shuffle_list=True):
        self.file_list=file_list
        self.batch_size=batch_size
        self.read_fn=read_fn
        self.cur_pos=0

        self.cur_data=None
        self.cur_data_pos=0
        self.cur_data_len=0
        self.cur_data_indices=None
        self.done=False

        self.shuffle_list=shuffle_list
        if self.shuffle_list: random.shuffle(self.file_list)

    def read_data(self):
        if self.cur_pos == len(self.file_list):
            self.cur_pos = 0
            if self.shuffle_list: random.shuffle(self.file_list)
            self.done=True
            return False

        self.cur_data = self.read_fn(self.file_list[self.cur_pos])
        self.cur_data_pos = 0
        self.cur_data_len = len(self.cur_data[0])
        self.cur_data_indices = range(self.cur_data_len)
        self.cur_pos+=1
        if self.shuffle_list: random.shuffle(self.cur_data_indices)
        return True

    def require_data(self,require_size):
        data_len=len(self.cur_data)
        cur_batch=[[] for _ in xrange(data_len)]
        actual_size=min(require_size+self.cur_data_pos,self.cur_data_len)-self.cur_data_pos
        indices=self.cur_data_indices[self.cur_data_pos:self.cur_data_pos+actual_size]
        for i in xrange(data_len):
            for idx in indices:
                cur_batch[i].append(self.cur_data[i][idx])
        self.cur_data_pos+=actual_size

        return cur_batch,require_size-actual_size

    def next(self):
        # read file
        if self.cur_data is None or self.cur_data_pos>=self.cur_data_len:
            if self.done or not self.read_data():
                self.done=False
                raise StopIteration

        cur_batch,left_size=self.require_data(self.batch_size)
        while left_size>0:
            if not self.read_data(): break
            new_batch,left_size=self.require_data(left_size)
            for i in xrange(len(self.cur_data)):
                cur_batch[i]+=new_batch[i]

        # randomly sample to batch size
        if len(cur_batch[0])<self.batch_size:
            indices=np.random.choice(len(cur_batch[0]),self.batch_size)
            new_batch=[[] for _ in xrange(len(self.cur_data))]
            for i in xrange(len(self.cur_data)):
                for idx in indices:
                    new_batch[i].append(cur_batch[i][idx])

            cur_batch=new_batch

        return cur_batch

    def __iter__(self): return self



def test_provider():
    def read_fn(num):
        return [range(num),range(num)]
    nums=np.random.randint(1,4,5)
    print nums
    provider=Provider(nums,4,read_fn,False)

    for val in provider:
        print val

if __name__=="__main__":
    test_provider()


