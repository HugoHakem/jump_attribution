# Name of your emacs binary
EMACS=emacs

BATCH=$(EMACS) --batch --no-init-file					\
  --eval '(require (quote org))'					\
  --eval "(org-babel-do-load-languages 'org-babel-load-languages	\
         '((shell . t)))"   						\
  --eval "(setq org-confirm-babel-evaluate nil)"			\
  --eval '(setq starter-kit-dir default-directory)'			\
  --eval '(org-babel-tangle-file "readme.org")'	              		\
  --eval '(org-babel-load-file   "readme.org")'

files_org  = $(wildcard poster*.org)
files_pdf  = $(addprefix pub/pdf/,$(files_org:.org=.pdf))

all: pdf 

pdf: $(files_pdf)
pub/pdf/%.pdf: %.org
	@$(BATCH) --visit "$<" --funcall org-publish-pdf
	@rm readme.el

clean:
	@rm -f *.elc *.aux *.tex *.pdf *~ *.sty
	@rm -rf pub
