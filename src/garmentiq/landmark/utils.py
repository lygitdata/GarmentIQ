import numpy as np


def find_instruction_landmark_index(instruction_landmarks: dict, predefined: bool):
  if predefined:
    return [int(k) - 1 for k, v in instruction_landmarks.items() if v.get('predefined') is True]
  else:
    return [int(k) - 1 for k, v in instruction_landmarks.items() if v.get('predefined') is False]

def fill_instruction_landmark_coordinate(instruction_landmarks: dict, index: list, fill_in_value: np.array):
  for k in instruction_landmarks:
      idx = int(k) - 1
      if idx in index:
          preds_idx = index.index(idx)
          instruction_landmarks[k]['x'] = float(fill_in_value[0, preds_idx, 0])
          instruction_landmarks[k]['y'] = float(fill_in_value[0, preds_idx, 1])
  return instruction_landmarks