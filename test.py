from concurrent.futures import ProcessPoolExecutor
import time

def foo(x): return x*x

w = []
l = []
x = 0
start_time = time.time()
with ProcessPoolExecutor() as executor: # parallelize the l sampling
        for i in range(50000):
                x = i
                if x % 5 == 0:
                        w.append(x)
                        future = executor.submit(foo, x)
                        l.append(future.result())
print(f"Before Parallel time: {time.time() - start_time}") # how ure supposed to run things

# w = []
# l = []
# x = 0
# start_time = time.time()
# for i in range(50000):
#         x = i
#         if x % 5 == 0:
#                 with ProcessPoolExecutor() as executor: # parallelize the l sampling
#                         w.append(x)
#                         future = executor.submit(foo, x)
#                         l.append(future.result())
# print(f"After Parallel time: {time.time() - start_time}") # doesn't work

# w = []
# l = []
# x = 0
# a = 3
# start_time = time.time()
# with ProcessPoolExecutor() as executor: # parallelize the l sampling
#         if a == 3:
#                 w = [i for i in range(50000)]
#                 for x in w:
#                         future = executor.submit(foo, x)
#                         l.append(future.result())
# print(f"Pos 0 Parallel time: {time.time() - start_time}")

# w = []
# l = []
# x = 0
# a = 3
# start_time = time.time()
# if a == 3:
#         w = [i for i in range(50000)]
#         with ProcessPoolExecutor() as executor: # parallelize the l sampling
#                 for x in w:
#                         future = executor.submit(foo, x)
#                         l.append(future.result())
# print(f"Pos 1 Parallel time: {time.time() - start_time}")


# w = []
# l = []
# x = 0
# start_time = time.time()
# for i in range(50000):
#         x = i
#         if x % 5 == 0:
#                 w.append(x)
#                 result = foo(x)
#                 l.append(result)
# print(f"Non-parallel time: {time.time() - start_time}")

# if self.reward >= 3: # w is sampled using approximation
# 	with ProcessPoolExecutor() as executor: # parallelize the l sampling
# 		w_samples = self.w_sampler(queries, self.dim, num_w_samples, seed=seed) #  approx the distribution into gaussian, and sample from using torch

# 		# start_time = time.time()
# 		# in parallel, sample l
# 		if self.lang <= 2: # use MC
# 			for i in range(num_w_samples):
# 				future = executor.submit(self.l_sampler, queries, w_samples[i], self.dim, num_l_samples_per_w * thin_l + burn_in_l, burn_in=burn_in_l, thin=thin_l, seed=seed + i)
# 				l_samples.append(future.result())
# 		else: # use sampling
# 			for i in range(num_w_samples):
# 				future = executor.submit(self.l_sampler, queries, w_samples[i], self.dim, num_l_samples_per_w, seed=seed + i)
# 				l_samples.append(future.result())

# 			print(f"Sampling l took: {time.time() - start_time}")

# elif self.reward <= 2: # w is sampled using mcmc
# 	with ProcessPoolExecutor() as executor: # parallelize the l sampling
# 		prev_w = initial_w
# 		for i in range(num_w_samples * thin_w + burn_in_w):
# 			w = self.w_sampler(queries, self.dim, prev_w, seed=seed) # do one step in mcmc to sample w
# 			prev_w = w

# 			# in parallel, sample l
# 			if i >= burn_in_w and i % thin_w == 0:
# 				if self.lang <= 2: # use MC
# 					w_samples.append(w)
# 					future = executor.submit(self.l_sampler, queries, w, self.dim, num_l_samples_per_w * thin_l + burn_in_l, burn_in=burn_in_l, thin=thin_l, seed=seed + i)
# 					l_samples.append(future.result())
# 				else: # use sampling
# 					w_samples.append(w)
# 					future = executor.submit(self.l_sampler, queries, w, self.dim, num_l_samples_per_w, seed=seed + i)
# 					l_samples.append(future.result())
# 		w_samples = torch.stack(w_samples)