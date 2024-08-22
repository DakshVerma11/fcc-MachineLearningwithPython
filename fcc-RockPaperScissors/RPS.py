# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.

def player(prev_play, opponent_history=[]):
    opponent_history.append(prev_play)
    last_ten = opponent_history[-10:]
    
    if len(opponent_history) < 2:
        return "R"
    
    most_frequent = max(set(last_ten), key=last_ten.count)
    if most_frequent == '':
        most_frequent = "R"

    ideal_response = {'P': 'S', 'R': 'P', 'S': 'R','':'P'}

    if len(last_ten) >= 2 and last_ten[-1] == last_ten[-2]:
        return ideal_response[last_ten[-1]]

    if len(opponent_history) >= 3 and opponent_history[-1] == opponent_history[-3]:
        return ideal_response[opponent_history[-2]]

    return ideal_response[most_frequent]
