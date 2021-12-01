import os
import sys
import math
import copy


def train(filepath):
	print("Training..")
	with open(filepath, "r", encoding="utf8") as f:
		lines = f.readlines()

	# Set of all unique tokens in file
	tokens = []
	# Nested dictionary to keep track of emission count
	# {tag: {token: count} }
	emission_count = {} 

	# Iterate through file to update tokens and emission_count
	for line in lines:
		line_split = line.strip().rsplit(" ", 1)
		if len(line_split) == 2:
			token = line_split[0]
			tag = line_split[1]

			if token not in tokens:
				tokens.append(token)

			if tag not in emission_count:
				nested_tag_dict = {}
			else:
				nested_tag_dict = emission_count[tag]
			if token not in nested_tag_dict:
				nested_tag_dict[token] = 1
			else:
				nested_tag_dict[token] += 1
			emission_count[tag] = nested_tag_dict

	return tokens, emission_count


def est_emission_param(emission_count, token, tag):
	tag_dict = emission_count[tag]

	a = tag_dict.get(token, 0)	# Returns 0 if none
	b = sum(tag_dict.values())

	return a / b


def est_emission_param(emission_count, token, tag, k=1):
	tag_dict = emission_count[tag]

	if token != "#UNK#":
		a = tag_dict.get(token, 0)
	else:
		a = k 
	b = sum(tag_dict.values()) + k

	return a / b


def get_sentence_tag(sentence, tokens, emission_count, k=1):
	pred_tags = []

	for word in sentence:
		pred_tag = ""
		max_emission = float('-inf')

		for tag in emission_count:
			if word not in tokens:
				word = "#UNK#"

			if word in emission_count[tag] or word == "#UNK#":
				emission = est_emission_param(emission_count, word, tag, k)
				if emission > max_emission:
					pred_tag = tag 
					max_emission = emission

		pred_tags.append(pred_tag)

	return pred_tags


def evaluate(filepath, tokens, emission_count, k=1):
	print("Evaluating..")
	with open(filepath, "r", encoding="utf8") as f:
		lines = f.readlines()

	all_pred_tags = []

	sentence = []
	for line in lines:
		if line != "\n":
			sentence.append(line.strip())
		else:
			pred_tags = get_sentence_tag(sentence, tokens, emission_count, k)
			all_pred_tags += pred_tags + ["\n"]

			sentence = []

	return lines, all_pred_tags


def write_output(filepath, lines, all_pred_tags):
	print("Writing output..")
	with open(filepath, "w", encoding="utf8") as f:
		for i in range(len(lines)):
			word = lines[i].strip()

			if word != "\n":
				tag = all_pred_tags[i]

				if tag != "\n":
					f.write(word + " " + tag)
				f.write("\n")

	print("Output successfully written!")


if __name__ == '__main__':
	root_dir = "./"
	datasets = ["ES", "RU"]

	for dataset in datasets:
		print("For dataset {}:".format(dataset))
		train_path = root_dir + "{}/train".format(dataset)
		evaluation_path = root_dir + "{}/dev.in".format(dataset)

		# Train
		tokens, emission_count = train(train_path)

		# Estimate emission parameters using MLE
		"""
		for tag in emission_count:
			for token in tokens:
				#emission = est_emission_param(emission_count, token, tag)		# Without tackling special word token
				emission = est_emission_param(emission_count, token, tag, k=1)	# With special word token
				if emission != 0:
					print("Emission parameters for {x} given {y}: {emission}".format(x=token, y=tag, emission=emission))
		"""
		
		# Evaluate
		lines, all_pred_tags = evaluate(evaluation_path, tokens, emission_count, k=1)

		# Write output file
		output_path = root_dir + "{}/dev.p1.out".format(dataset)
		write_output(output_path, lines, all_pred_tags)

		print("Dataset {} done.".format(dataset))

	print("Done for all datasets!!")