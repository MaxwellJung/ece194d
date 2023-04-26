d1 = 1
d2 = 5
d3 = 10
d4 = 25
denominations = [d1, d2, d3, d4]

state = None
action = None

def greedy_policy(s):
    return max([d for d in denominations if d <= s])

def reward(s, a):
    return 0

def coins(cents):
    coin_list = []
    state = cents
    while(state > 0):
        action = greedy_policy(state)
        coin_list.append(action)
        state -= action
        
    return coin_list

def main():
    goal = 17
    print(f'You need {coins(goal)} to make up {goal} cents.')

if __name__ == '__main__':
    main()