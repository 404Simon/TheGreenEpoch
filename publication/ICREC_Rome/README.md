# TheGreenEpochPaper

LNCS (Springer) paper source, edited collaboratively via GitHub and Overleaf.

## Layout

```
main.tex          # the paper (main document, also in Overleaf)
references.bib    # bibliography
assets/           # figures: SVG from the simulation output + converted EPS
Makefile          # build + figure pipeline
llncs.cls splncs04.bst llncsdoc.pdf readme.txt history.txt fig1.eps
                  # unmodified Springer LNCS package — do not edit
```

## Building

```
make            # build main.pdf
make svgs       # regenerate figure SVGs from the simulation results
make clean      # remove build artifacts and generated figures
```

Uses `latexmk -pdfps` (latex → dvi → ps → pdf) because the figures are EPS; needs
`rsvg-convert`. Build artifacts are gitignored. `assets/` **is** committed — Overleaf
cannot run the Makefile, so the figures must be in the repo.

Overleaf needs no configuration: main document is `main.tex`, and `llncs.cls` /
`splncs04.bst` are picked up from the project root.

## Workflow

GitHub is the source of truth. Overleaf is a mirror, synced **manually in both
directions** — nothing is automatic.

**Git users** — pull, edit, commit, then push to both:

```
git push origin main
git push overleaf main
```

**Overleaf users** need no Git access: edit in the browser, auto-save, use comments
or track changes for discussion. The maintainer merges those edits back:

```
git fetch overleaf
git --no-pager diff --name-status main..overleaf/main   # review
git merge overleaf/main
git push origin main
```

If `git push overleaf main` is rejected, Overleaf has unmerged edits — run the merge
above first, then push again.

**First-time setup (once per clone)** — a fresh clone has only `origin`:

```
git remote add overleaf https://git.overleaf.com/6a58d6e46c66edeec9d89320
git config --global credential.helper store
git fetch overleaf    # username "git", password = Overleaf Git token
```

Generate the token at Overleaf → *Account Settings → Git Integration*. It is stored
in `~/.git-credentials` — never commit it. Verify with `git branch -r | grep overleaf`.

## Rules

- `main` is the only shared branch — no force-push, no rebase
- Always pull before editing; one person per file at a time
- Split into multiple `.tex` files if the paper grows
