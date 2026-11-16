# Third-party notices

The root Apache-2.0 license applies to original JACE code. It does not relicense
third-party models, source files or datasets.

- **SASRec / LIME-Rec.** `src/jace_rec/sasrec.py` and expert-side scoring derive
  from [LIME-Rec](https://github.com/Double-wk/LIME-Rec), an Apache-2.0 project.
  Its license is retained in `vendor/LICENSE-LIME-Rec-APACHE-2.0`. This is the
  checkpoint-compatible PyTorch expert, not the canonical TensorFlow SASRec
  baseline. Original SASRec: [kang205/SASRec](https://github.com/kang205/SASRec).
- **GRAM / FiD / T5.** The GRAM integration targets
  [skleee/GRAM](https://github.com/skleee/GRAM) and
  [facebookresearch/FiD](https://github.com/facebookresearch/FiD).
  Backend files are fetched directly from the author's repository at commit
  `7ac4d9272a57beed9df35c27ea34221f6e4a8fb1`; they are not included in this
  source repository or the JACE asset archives. The installer verifies source
  hashes, removes unused IPython imports, and uses reshape in the FiD wrapper.
  Original source notices remain intact. Upstream terms continue to apply;
  the JACE license does not grant additional rights over downloaded sources.
- **ActionPiece.** The optional adapter connects to the author's implementation
  at [google-deepmind/action_piece](https://github.com/google-deepmind/action_piece),
  PR #6 commit `9dce2075834cf034880228135e52e689fd43cf34`.
  Upstream model, tokenizer and training code are not bundled.
- **APAO.** The optional adapter targets
  [yuyq18/APAO](https://github.com/yuyq18/APAO), commit
  `9ab195920d541debcff80b255becb9dcbd44d6f7`. Upstream code and assets are not bundled.
- **Pretrained models.** T5-small revision
  `df1b051c49625cf57a3d0d8d3863ed4d13564fe4` and Qwen3-Embedding-4B revision
  `5cf2132abc99cad020ac570b19d031efec650f2b` are external dependencies.
  Their provider terms apply to model and tokenizer files.
- **Data.** Amazon Reviews and Yelp data, processed sequences, metadata and
  derived vectors retain their source terms. The released inputs use GRAM's
  [preprocessed datasets](https://github.com/skleee/GRAM/tree/7ac4d9272a57beed9df35c27ea34221f6e4a8fb1/rec_datasets).
  Raw sequence/text files match that pinned source byte for byte. JACE adds
  the frozen event split, clean prefixes, expert embeddings and selected
  checkpoints. The source-code Apache license does not relicense the data.

PyTorch, Transformers, Tokenizers and NumPy retain their respective licenses.
