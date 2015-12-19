import theano
import theano.tensor as T

k = T.iscalar("k")
A = T.vector("A")

# Symbolic description of the result
# scan will provide areguments to fn in the following order:
# 1. sequences
# 2. output_info
# 3. non_sequences
# in this case sequences are not defined, then prior_result will be an output_info
# initially set to a vector of ones (same length as A), that will be used as an
# accumulator in the recurrece.
# The 2nd arg of fn will be a non_sequences A that has constant value, that means
# it doesn't change over the recurrence.
result, updates = theano.scan(fn=lambda prior_result, A: prior_result * A,
                              outputs_info=T.ones_like(A),
                              non_sequences=A,
                              n_steps=k)

# We only care about A**k, but scan has provided us with A**1 through A**k.
# Discard the values that we don't care about. Scan is smart enough to
# notice this and not waste memory saving them.
final_result = result[-1]

# compiled function that returns A**k
power = theano.function(inputs=[A,k], outputs=final_result, updates=updates)
#power = theano.function(inputs=[A,k], outputs=final_result)

print power(range(10),2)
print power(range(10),4)

