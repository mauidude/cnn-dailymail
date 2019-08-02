import sys
import csv
import os
import hashlib
import struct
import subprocess
import collections
import tensorflow as tf
from tensorflow.core.example import example_pb2

webis_raw = '/home/aditihegde/webis-rawdata/'
stories_dir = '../webis'
tokenized_stories_dir = '../webis-tokenized'

dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

finished_files_dir = "finished_files"
chunks_dir = os.path.join(finished_files_dir, "chunked")
out_file = "../webis_bin/test_webis.bin"

CHUNK_SIZE = 1000

os.environ['CLASSPATH'] ='../corenlp/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar'

def convert_to_stories(webis_raw, story_dir)
  files = os.listdir(webis_raw)
  count = 0
  for _file in files:
      with open(os.path.join(webis_raw, _file) as csv_file:
          csv_reader = csv.reader(csv_file, delimiter=',')
          for row in csv_reader:            
              f = open(stories_dir+'/webis-'+str(count)+".story", "w") 
              f.write(f'{row[0]}\n@highlight\n{row[1]}')     
              f.close()   
              count += 1
              
            
def read_text_file(text_file):
  lines = []
  with open(text_file, "r") as f:
    for line in f:
      lines.append(line.strip())
  return lines

def fix_missing_period(line):
  """Adds a period to a line that is missing a period"""
  if "@highlight" in line: return line
  if line=="": return line
  if line[-1] in END_TOKENS: return line
  # print line[-1]
  return line + " ."

def get_art_abs(story_file):
  lines = read_text_file(story_file)

  # Lowercase everything
  lines = [line.lower() for line in lines]

  # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
  lines = [fix_missing_period(line) for line in lines]

  # Separate out article and abstract sentences
  article_lines = []
  highlights = []
  next_is_highlight = False
  for idx,line in enumerate(lines):
    if line == "":
      continue # empty line
    elif line.startswith("@highlight"):
      next_is_highlight = True
    elif next_is_highlight:
      highlights.append(line)
    else:
      article_lines.append(line)

  # Make article into a single string
  article = ' '.join(article_lines)

  # Make abstract into a signle string, putting <s> and </s> tags around the sentences
  abstract = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in highlights])

  return article, abstract

def tokenize_stories(stories_dir, tokenized_stories_dir):
  """Maps a whole directory of .story files to a tokenized version using Stanford CoreNLP Tokenizer"""
  print("Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir))
  stories = os.listdir(stories_dir)
  # make IO list file
  print("Making list of files to tokenize...")
  with open("mapping.txt", "w") as f:
    for s in stories:
      f.write("%s \t %s\n" % (os.path.join(stories_dir, s), os.path.join(tokenized_stories_dir, s)))
  command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', 'mapping.txt']
  print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir))
  subprocess.call(command)
  print("Stanford CoreNLP Tokenizer has finished.")
  #os.remove("mapping.txt")

  # Check that the tokenized stories directory contains the same number of files as the original directory
  num_orig = len(os.listdir(stories_dir))
  num_tokenized = len(os.listdir(tokenized_stories_dir))
  if num_orig != num_tokenized:
    raise Exception("The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
  print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir))

def write_to_bin(tokenized_story_dir, out_file):
    """Reads the tokenized .story files and writes them to a out_file."""
  
    story_fnames = os.listdir(tokenized_story_dir)
    num_stories = len(story_fnames)

    with open(out_file, 'wb') as writer:
        for idx,s in enumerate(story_fnames):
          if idx % 1000 == 0:
            print("Writing story %i of %i; %.2f percent done" % (idx, num_stories, float(idx)*100.0/float(num_stories)))

      # Look in the tokenized story dirs to find the .story file
          story_file = os.path.join(tokenized_story_dir, s)
      
      # Get the strings to write to .bin file
          article, abstract = get_art_abs(story_file)

      # Write to tf.Example
          tf_example = example_pb2.Example()
          tf_example.features.feature['article'].bytes_list.value.extend([article.encode()])
          tf_example.features.feature['abstract'].bytes_list.value.extend([abstract.encode()])
          tf_example_str = tf_example.SerializeToString()
          str_len = len(tf_example_str)
          writer.write(struct.pack('q', str_len))
          writer.write(struct.pack('%ds' % str_len, tf_example_str))

    print("Finished writing file %s\n" % out_file)
                
def chunk_file(set_name):
  in_file = set_name+'.bin'
  reader = open(in_file, "rb")
  chunk = 0
  finished = False
  while not finished:
    chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk)) # new chunk
    with open(chunk_fname, 'wb') as writer:
      for _ in range(CHUNK_SIZE):
        len_bytes = reader.read(8)
        if not len_bytes:
          finished = True
          break
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, example_str))
      chunk += 1


def chunk_all():
  # Make a dir to hold the chunks
  if not os.path.isdir(chunks_dir):
    os.mkdir(chunks_dir)
  # Chunk the data
  chunk_file('../webis_bin/test_webis')
  print("Saved chunked data in %s" % chunks_dir)


convert_to_stories(webis_raw, stories_dir)
tokenize_stories(stories_dir, tokenized_stories_dir)
write_to_bin(tokenized_story_dir, out_file)
chunk_all()
