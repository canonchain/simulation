import random
import hashlib
import sys
# Clock offset
CLOCKOFFSET = 1
# Block time
BLKTIME = 40
# Round run time
ROUND_RUNTIME = 500000
# Number of rounds
ROUNDS = 2000
# Block reward
BLKREWARD = 1
# Reward for including x ticks' worth of transactions
# Linear by default, but sublinear formulas are
# probably most accurate
get_txreward = lambda ticks: 0.0001 * ticks
# Latency function
latency_sample = lambda L: int((random.expovariate(1) * ((L/1.33)**0.75))**1.33)
# latency = lambda: random.randrange(15) if random.randrange(10) else 47
# Offline rate
OFFLINE_RATE = 0.03

BLANK_STATE = {'transactions': 0}

class Simulation():
    def __init__(self, validators):
        for i, v in enumerate(validators):
            v.id = i
            v.simulation = self
        self.validators = validators
        self.time = 0
        self.next_id = 0
        self.gvcache = {}

    def run(self, rounds):
        # Run the simulation
        for i in range(rounds):
            for m in self.validators:
                m.mine()
                m.listen()
            self.time += 1
            if i % (rounds // 100) == 0:
                print 'Completed %d rounds out of %d' % (i, rounds)
    
    def get_validator(self, randao, skips):
        key = (randao << 32) + skips
        if key not in self.gvcache:
            self.gvcache[key] = sha256_as_int(key) % len(self.validators)
        return self.gvcache[key]


class Block():
    def __init__(self, parent, state, maker, number=0, skips=0):
        self.prevhash = parent.hash if parent else 0
        self.state = state
        self.number = number
        self.height = parent.height + 1 if parent else 0
        self.hash = random.randrange(10**20) + 10**23 * self.height
        self.randao = sha256_as_int((parent.randao if parent else 0) + maker)
        self.totskips = (parent.totskips if parent else 0) + skips

def sha256_as_int(v):
    hashbytes = hashlib.sha256(str(v)).digest()[:4]
    o = 0
    for b in hashbytes:
        o = (o << 8) + ord(b)
    return o

GENESIS = Block(None, BLANK_STATE, 0)

# Insert a key/value pair into a state
# This is abstracted away into a method to make it easier to
# swap the state out with an immutable dict library or whatever
# else to increase efficiency
def update_state(s, k, v):
    s2 = {_k: _v for _k, _v in s.items()}
    s2[k] = v
    return s2

# Get a key from a state, default zero
def get_state(s, k):
    return s.get(k, 0)


class Validator():
    def __init__(self, strategy, latency=5):
        # The block that the validator considers to be the head
        self.head = GENESIS
        # A map of tick -> blocks that the validator will receive
        # during that tick
        self.listen_queue = {}
        # Blocks that the validator knows about
        self.blocks = {GENESIS.hash: GENESIS}
        # Scores (~= total subtree weight) for those blocks
        self.scores = {GENESIS.hash: 0}
        # When the validator received each block
        self.time_received = {GENESIS.hash: 0}
        # Received too early?
        self.received_too_early = {}
        # Blocks with missing parents, mapping parent hash -> list
        self.orphans_by_parent = {}
        # ... mapping hash -> list
        self.orphans = {}
        # This validator's clock is off by this number of ticks
        self.time_offset = random.randrange(CLOCKOFFSET) - CLOCKOFFSET // 2
        # Set the validator's strategy
        self.set_strategy(strategy)
        # Keeps track of the highest block number a validator has already
        # produced a block at
        self.min_number = 0
        # The validator's ID
        self.id = None
        # Blocks created
        self.created = 0
        # Number of blocks to backtrack for GHOST
        self.backtrack = 40
        # The simulation that this validator is in
        self.simulation = None
        # Maximum number of skips to try
        self.max_skips = 8
        # Network latency
        self.latency = latency

    def set_strategy(self, strategy):
        # The number of ticks a validator waits before producing a block
        self.produce_delay = strategy[0]
        # The number of extra ticks a validator waits per skip (ie.
        # if you skip two validator slots then wait this number of ticks
        # times two) before producing a block
        self.per_block_produce_delay = strategy[1]
        # The number of extra ticks a validator waits per skip before
        # accpeint a block
        self.per_block_accept_delay = strategy[2]
        

    # Get the time from the validator's point of view
    def get_time(self):
        return max(self.simulation.time + self.time_offset, 0)

    # Add a block to the listen queue at the given time
    def add_to_listen_queue(self, time, obj):
        if time not in self.listen_queue:
            self.listen_queue[time] = []
        self.listen_queue[time].append(obj)

    def earliest_block_time(self, parent, skips):
        return 20 + parent.number * 20 + skips * 40

    def mine(self):
        # Is it time to produce a block?
        t = self.get_time()
        skips = 0
        head = self.head
        while self.simulation.get_validator(head.randao, skips) != self.id and skips < self.max_skips:
            skips += 1
        if skips == self.max_skips:
            return
        # If it is...
        if t >= self.time_received[head.hash] + self.produce_delay + self.per_block_produce_delay * skips \
                and head.number >= self.min_number:
            # Can't produce a block at this height anymore
            self.min_number = head.number + 1 + skips
            # Small chance to be offline
            if random.random() < OFFLINE_RATE:
                return
            # Compute my block reward
            my_reward = BLKREWARD + get_txreward(self.simulation.time - head.state['transactions'])
            # Claim the reward from the transactions since the parent
            new_state = update_state(head.state, 'transactions', self.simulation.time)
            # Apply the block reward
            new_state = update_state(new_state, self.id, get_state(new_state, self.id) + my_reward)
            # Create the block
            b = Block(head, new_state, self.id, number=head.number + 1 + skips, skips=skips)
            self.created += 1
            # print '---------> Validator %d makes block with hash %d and parent %d (%d skips) at time %d' % (self.id, b.hash, b.prevhash, skips, self.simulation.time)
            # Broadcast it
            for validator in self.simulation.validators:
                recv_time = self.simulation.time + 1 + latency_sample(self.latency + validator.latency)
                # print 'broadcasting, delay %d' % (recv_time - t)
                validator.add_to_listen_queue(recv_time, b)

    # If a validator realizes that it "should" have a block but doesn't,
    # it can use this method to request it from the network
    def request_block(self, hash):
        for validator in self.simulation.validators:
            if hash in validator.blocks:
                recv_time = self.simulation.time + 1 + latency_sample(self.latency + validator.latency)
                self.add_to_listen_queue(recv_time, validator.blocks[hash])

    # Process all blocks that it should receive during the current tick
    def listen(self):
        head = self.blocks[self.main_chain[-1]]
        if self.simulation.time in self.listen_queue:
            for blk in self.listen_queue[self.simulation.time]:
                self.accept_block(blk)
        if self.simulation.time in self.listen_queue:
            del self.listen_queue[self.simulation.time]

    def get_score_addition(self, blk):
        parent = self.blocks[blk.prevhash]
        skips = blk.number - parent.number - 1
        return (0 if blk.hash in self.received_too_early else 10**20) + random.randrange(100) - 50

    def accept_block(self, blk):
        t = self.get_time()
        # Parent not found or at least not yet processed
        if blk.prevhash not in self.blocks and blk.hash not in self.orphans:
            self.request_block(blk.prevhash)
            if blk.prevhash not in self.orphans_by_parent:
                self.orphans_by_parent[blk.prevhash] = []
            self.orphans_by_parent[blk.prevhash].append(blk.hash)
            self.orphans[blk.hash] = blk
            # print 'validator %d skipping block %d: parent %d not found' % (self.id, blk.hash, blk.prevhash)
            return
        # Already processed?
        if blk.hash in self.blocks or blk.hash in self.orphans:
            # print 'validator %d skipping block %d: already processed' % (self.id, blk.hash)
            return
        # Too early? Re-append at earliest allowed time
        parent = self.blocks[blk.prevhash]
        skips = blk.number - parent.number - 1
        alotted_recv_time = self.time_received[parent.hash] + skips * self.per_block_accept_delay
        if t < alotted_recv_time:
            self.received_too_early[blk.hash] = alotted_recv_time - t
            self.add_to_listen_queue((alotted_recv_time - t) * 2 + self.simulation.time, blk)
            # print 'too early, validator %d delaying %d (%d vs %d)' % (self.id, blk.hash, t, alotted_recv_time)
            return
        # Add the block and compute the score
        # print 'Validator %d receives block, hash %d, time %d' % (self.id, blk.hash, self.simulation.time)
        self.blocks[blk.hash] = blk
        self.time_received[blk.hash] = t
        if blk.hash in self.orphans:
            del self.orphans[blk.hash]
        # Process the scoring rule
        self.head
        # print 'post', self.main_chain
        # self.scores[blk.hash] = self.scores[blk.prevhash] + get_score_addition(skips)
        if self.orphans_by_parent.get(blk.hash, []):
            for c in self.orphans_by_parent[blk.hash]:
                # print 'including previously rejected child of %d: %d' % (blk.hash, c)
                b = self.orphans[c]
                del self.orphans[c]
                self.accept_block(b)
            del self.orphans_by_parent[blk.hash]


def simple_test(baseline=[10, 40, 31]):
    # Define the strategies of the validators
    strategy_groups = [
        #((time before publishing a block, time per skip to wait before producing a block, time per skip to wait before accepting), latency, number of validators with this strategy)
        # ((baseline[0], baseline[1], baseline[2]), 12, 4),
        # ((baseline[0], baseline[1], baseline[2]), 11, 4),
        # ((baseline[0], baseline[1], baseline[2]), 10, 4),
        # ((baseline[0], baseline[1], baseline[2]), 9, 4),
        # ((baseline[0], baseline[1], baseline[2]), 8, 4),
        # ((baseline[0], baseline[1], baseline[2]), 7, 4),
        ((baseline[0], baseline[1], baseline[2]), 6, 5),
        ((baseline[0], baseline[1], baseline[2]), 5, 5),
        ((baseline[0], baseline[1], baseline[2]), 4, 5),
        ((baseline[0], baseline[1], baseline[2]), 3, 5),
        ((baseline[0], baseline[1], baseline[2]), 2, 5),
        ((baseline[0], baseline[1], baseline[2]), 1, 5),
        # ((baseline[0], int(baseline[1] * 0.33), baseline[2]), 3, 2),
        # ((baseline[0], int(baseline[1] * 0.67), baseline[2]), 3, 2),
        # ((baseline[0], int(baseline[1] * 1.5), baseline[2]), 3, 2),
        # ((baseline[0], int(baseline[1] * 2), baseline[2]), 3, 2),
        # ((baseline[0], baseline[1], int(baseline[2] * 0.33)), 3, 2),
        # ((baseline[0], baseline[1], int(baseline[2] * 0.67)), 3, 2),
        # ((baseline[0], baseline[1], int(baseline[2] * 1.5)), 3, 2),
        # ((baseline[0], baseline[1], int(baseline[2] * 2)), 3, 2),
    ]
    sgstarts = [0]
    
    validators = []
    for s, l, c in strategy_groups:
        sgstarts.append(sgstarts[-1] + c)
        for i in range(c):
            validators.append(Validator(s, l))
    
    Simulation(validators).run(ROUND_RUNTIME)
    
    def report(validators):
        head = validators[0].blocks[validators[0].main_chain[-1]]
        
        print 'Head block number:', head.number
        print 'Head block height:', head.height
        print head.state
        # print validators[0].scores
        
        for i, ((s, l, c), pos) in enumerate(zip(strategy_groups, sgstarts)):
            totrev = 0
            totcre = 0
            for j in range(pos, pos + c):
                totrev += head.state.get(j, 0)
                totcre += validators[j].created
            print 'Strategy group %d: average %d / %d / %d' % (i, totrev * 1.0 / c, totcre * 1.0 / c, (totrev * 2.0 - totcre) / c)

    report(validators)

def evo_test(initial_s=[1, 40, 27]):
    s0 = [20, 40, 40]
    s = initial_s
    INITIAL_GROUP = 20
    DEVIATION_GROUP = 2
    LATENCY = 3
    for i in range(ROUNDS):
        print 's:', s, ', starting round', i
        strategy_groups = [
            (s0, LATENCY, 1),
            (s, LATENCY, INITIAL_GROUP - 1),
        ]
        for j in range(len(s)):
            t = [x for x in s]
            t[j] = int(t[j] * 1.3)
            strategy_groups.append((t, LATENCY, DEVIATION_GROUP))
            u = [x for x in s]
            u[j] = int(u[j] * 0.7)
            strategy_groups.append((u, LATENCY, DEVIATION_GROUP))

        sgstarts = [0]
        
        validators = []
        for _s, _l, c in strategy_groups:
            sgstarts.append(sgstarts[-1] + c)
            for i in range(c):
                validators.append(Validator(_s, _l))

        Simulation(validators).run(ROUND_RUNTIME)
        head = validators[0].blocks[validators[0].main_chain[-1]]
        base_instate = sum([head.state.get(j, 0) for j in range(1, INITIAL_GROUP)]) * 1.0 / (INITIAL_GROUP - 1)
        base_totcreated = sum([validators[j].created for j in range(1, INITIAL_GROUP)]) * 1.0 / (INITIAL_GROUP - 1)
        base_reward = base_instate * 2 - base_totcreated
        print 'old s:', s
        for j in range(len(s)):
            L = INITIAL_GROUP + DEVIATION_GROUP * 2 * j
            M = INITIAL_GROUP + DEVIATION_GROUP * (2 * j + 1)
            R = INITIAL_GROUP + DEVIATION_GROUP * (2 * j + 2)
            up_instate = sum([head.state.get(k, 0) for k in range(L, M)]) * 1.0 / (M - L)
            up_totcreated = sum([validators[k].created for k in range(L, M)]) * 1.0 / (M - L)
            up_reward = up_instate * 2 - up_totcreated
            down_instate = sum([head.state.get(k, 0) for k in range(M, R)]) * 1.0 / (R - M)
            down_totcreated = sum([validators[k].created for k in range(M, R)]) * 1.0 / (R - M)
            down_reward = down_instate * 2 - down_totcreated
            print 'Adjusting variable %d: %.3f %.3f %.3f' % (j, down_reward, base_reward, up_reward)
            if up_reward > base_reward > down_reward:
                print 'increasing', j, s
                s[j] = int(s[j] * min(1 + (up_reward - base_reward) * 2. / base_reward, 1.2) + 0.49)
            elif down_reward > base_reward > up_reward:
                print 'decreasing', j, s
                s[j] = int(s[j] * max(1 - (down_reward - base_reward) * 2. / base_reward, 0.8) + 0.49)
        print 'new s:', s

if len(sys.argv) >= 2 and sys.argv[1] == "evo":
    if len(sys.argv) > 4:
        evo_test([int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])])
    else:
        evo_test()
elif len(sys.argv) >= 2 and sys.argv[1] == "onetime":
    if len(sys.argv) > 4:
        simple_test([int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])])
    else:
        simple_test()
else:
    print 'Use evo or onetime'
