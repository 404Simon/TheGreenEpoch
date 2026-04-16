%% i7-thesis.cls (c) 2012 David Eckhoff <de@cs.fau.de>
%% based on ccs-thesis.cls by
%% Copyright (C) 2009-2011 Christoph Sommer <christoph.sommer@uibk.ac.at>
%%                         Falko Dressler <falko.dressler@uibk.ac.at>
%% Modifications:
%%               2018      Kai-Steffen Hielscher <kai-steffen.hielscher@fau.de>
%%               2021      Joerg Deutschmann <joerg.deutschmann@fau.de> removed babeltools
%%               2022      Sebastian Endres <basti.endres@fau.de>
%%               2022	   Matthias Schäfer <matthias.schaefer@fau.de>
%%               2025      Jonathan Fellerer <jonathan.fellerer@fau.de>
%%               2026      Martin Ottens <martin.ottens@fau.de>

%
% License (pick one or both):
% - Creative Commons Attribution-Share Alike 3.0 (or later)
% - GPL v2 (or later)
%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{i7-thesis}[2025/01/14 The i7 Thesis Class]  % YYYY/MM/DD


% -----------------------------------------------------------------------------
% Class options
% -----------------------------------------------------------------------------
\RequirePackage{ifthen}

% [germanthesis] - Thesis is written in German
\newboolean{germanthesis}
\setboolean{germanthesis}{false}
\DeclareOption{germanthesis}{\setboolean{germanthesis}{true}}

% [plainunnumbered] - Don't print numbers on plain pages
\newboolean{plainunnumbered}
\setboolean{plainunnumbered}{false}
\DeclareOption{plainunnumbered}{\setboolean{plainunnumbered}{true}}

% [earlydraft] - Settings for quick draft printouts
\newboolean{earlydraft}
\setboolean{earlydraft}{false}
\DeclareOption{earlydraft}{\setboolean{earlydraft}{true}}

% [watermark] - Print current time/date at bottom of each page
\newboolean{watermark}
\setboolean{watermark}{false}
\DeclareOption{watermark}{\setboolean{watermark}{true}}

% [phdthesis] - switch to PhD thesis style
\newboolean{phdthesis}
\setboolean{phdthesis}{false}
\DeclareOption{phdthesis}{\setboolean{phdthesis}{true}}

% [modernchapter] - modern chapter marks
\newboolean{modernchapter}
\setboolean{modernchapter}{false}
\DeclareOption{modernchapter}{\setboolean{modernchapter}{true}}

% [moderntitle] - modern title page
\newboolean{moderntitle}
\setboolean{moderntitle}{false}
\DeclareOption{moderntitle}{\setboolean{moderntitle}{true}}

% [twoside] - double sided
\newboolean{oneside}
\setboolean{oneside}{true}
\DeclareOption{twoside}{\setboolean{oneside}{false}}

% [privacy] - Generate a more privacy aware pdf
\newboolean{privacy}
\setboolean{privacy}{false}
\DeclareOption{privacy}{\setboolean{privacy}{true}}

% \newboolean{gnuplot}
% \setboolean{gnuplot}{false}
% \DeclareOption{gnuplot}{\setboolean{gnuplot}{true}}

\newboolean{cscoversheet}
\setboolean{cscoversheet}{false}
\DeclareOption{cscoversheet}{\setboolean{cscoversheet}{true}}

\newboolean{colorcoversheet}
\setboolean{colorcoversheet}{false}
\DeclareOption{colorcoversheet}{\setboolean{colorcoversheet}{true}}

\ProcessOptions \relax
% -----------------------------------------------------------------------------


% -----------------------------------------------------------------------------
% Basics
% -----------------------------------------------------------------------------
% This class is based on report
\ifthenelse{\boolean{oneside}}{
\LoadClass[a4paper,10pt,onecolumn,oneside,openright]{report}
\RequirePackage[a4paper,textwidth=345pt,textheight=598pt,bindingoffset=0cm,marginparwidth=3cm,hmarginratio=1:1,vmarginratio=1:1]{geometry}
}{
\LoadClass[a4paper,10pt,onecolumn,twoside,openright]{report}
\RequirePackage[a4paper,textwidth=345pt,textheight=598pt,bindingoffset=0cm,marginparwidth=3cm,hmarginratio=1:1,vmarginratio=1:1]{geometry}
}

% 150 percent line spacing
\RequirePackage{setspace}
\onehalfspacing

% Allow more (and larger) floats on text pages
\renewcommand{\topfraction}{0.9}  % use up to ..% of space on top
\renewcommand{\bottomfraction}{0.8}  % use up to ..% of space on bottom
\setcounter{topnumber}{4} % place up to .. on top
\setcounter{bottomnumber}{1} % place up to .. on bottom
\setcounter{totalnumber}{4} % place up to .. total
\renewcommand{\textfraction}{0.07} % allow down to ..% of text
\renewcommand{\floatpagefraction}{0.7} % fill at least ..% of float pages (must be less than \topfraction)

% Avoid widows and orphans (single line on bottom or top of page, respectively) at almost any cost
\clubpenalty = 10000
\widowpenalty = 10000
%\raggedbottom

% Default fonts
\newcommand{\mypageheadfont}{\normalfont}
\newcommand{\myheadingfont}{\normalfont}
%\RequirePackage[osf]{libertine}  % Linux Libertine
%\renewcommand{\mypageheadfont}{\normalfont\libertine}
%\renewcommand{\myheadingfont}{\normalfont\libertine}
\RequirePackage{lmodern}  % TT: Latin Modern
\RequirePackage{helvet}  % SF: Helvetica
\RequirePackage[bitstream-charter,sfscaled=false]{mathdesign}  % RM: Bitstream Charter
% TODO: nicer math font?

% More encoding and typesetting fixes and tweaks
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage{textcomp}
\RequirePackage[babel,protrusion=true,expansion=true,tracking=false,kerning=true,spacing=true]{microtype}
\newcommand{\mydisableprotrusion}{\microtypesetup{protrusion=false}}
\RequirePackage{hyphenat}

% Graphics insertion and filetypes' order of precedence
\RequirePackage[pdftex]{graphicx}
\pdfminorversion=6
\DeclareGraphicsRule{.out.png}{png}{.out.png}{}
\ifthenelse{\boolean{earlydraft}}{
	\DeclareGraphicsExtensions{.thumb.jpg,.out.png,.pdf,.png,.jpg}
	\DeclareGraphicsRule{.thumb.jpg}{jpg}{.thumb.jpg}{}
}{
	\DeclareGraphicsExtensions{.out.png,.pdf,.png,.jpg}
}

% Typesetting URLs
\RequirePackage[lowtilde]{url}
\urlstyle{tt}

% For hyperref<->algorithm interaction
\RequirePackage{float}

% Cross-references are clickable
\RequirePackage[pdftex,
				bookmarks=true,
				bookmarksnumbered=true,
				colorlinks=false,
				pdfdisplaydoctitle=true,
				breaklinks=true]{hyperref}

% Captions and Subfigures
\RequirePackage[font=small, labelfont=bf, labelsep=endash, margin=1cm]{caption}
\RequirePackage{subcaption}
\captionsetup{compatibility=false}

% Citations
\RequirePackage[backend=biber, style=ieee, dashed=false]{biblatex}
\addbibresource{bibliography.bib}

% \enquote
\RequirePackage{csquotes}
\RequirePackage{gensymb}

% The "qty", "num" and "unit" commands for typesetting units
\RequirePackage[%
	%binary-units,
	% range-phrase={--},
	range-units=single,
	per-mode=symbol,
	detect-all,
	load-configurations=binary,
	forbid-literal-units,
	exponent-product=\cdot,
	output-product=\cdot,
]{siunitx}
\DeclareSIUnit\month{month}
\DeclareSIUnit\year{yr}
\DeclareSIUnit{\EUR}{\text{\euro}}
\DeclareSIUnit{\CENT}{\text{ct}}
\catcode`\%=12\relax
\DeclareSIUnit[number-unit-product=]\percent{\,%}
\catcode`\%=14\relax

% Commands for beautiful tables
\RequirePackage{booktabs}
\RequirePackage{tabularx}

% Equation typesetting fixes and tweaks + proof/theorem environment
\RequirePackage{amsmath}
\RequirePackage{mathtools} % for 'dcases*' env.
\RequirePackage{amsthm}
\newtheorem{thm}{Theorem}[section]

% Nomenclature package for easier math equations
\RequirePackage{nomencl}
\RequirePackage{etoolbox}  % to group nomenclature entries
\makenomenclature
\NewDocumentCommand\nomencldef{mmmO{}}{
	\newcounter{COUNT_#1}
	\expandafter\DeclareMathOperator\csname #1\endcsname{#2\stepcounter{COUNT_#1}}
	\AtEndDocument{%
		\ifthenelse{\value{COUNT_#1}>0}{% this ensures, that only used nomenclature is included
			\nomenclature[#4]{\(#2\)}{#3}
		}{}
	}
}
% -----------------------------------------------------------------------------


% -----------------------------------------------------------------------------
% Internationalization support
% -----------------------------------------------------------------------------

% Multi-language support using Babel
% Update Joerg Deutschmann 2021-02-05: removed babeltools which has been deprecated
% Added [shorthands=off] option the babel package (i.e. "a will not try to produce an umlaut)
% removed [shorthands=off] to allow explicit hyphenation
\ifthenelse{\boolean{germanthesis}}{
	\RequirePackage[american,ngerman]{babel}
}{
	\RequirePackage[ngerman,american]{babel}
}

% % Don't indent paragraphs in german theses. Instead, leave blank space between paragraphs.
% \ifthenelse{\boolean{germanthesis}}{
% 	\RequirePackage{parskip}
% }{
% }

% Time and date functions
\RequirePackage[useregional]{datetime2}

% Clever References
\ifthenelse{\boolean{germanthesis}}{
	\RequirePackage[ngerman]{cleveref}[2009/17/04]
	\def\cref{\Cref}
	\def\crefrange{\Crefrange}
	\crefname{lstlisting}{Listing}{Listings}
	\Crefname{lstlisting}{Listing}{Listings}
	\crefname{listing}{Listing}{Listings}
	\Crefname{listing}{Listing}{Listings}
	\crefname{algorithm}{Algorithmus}{Algorithmen}
	\Crefname{algorithm}{Algorithmus}{Algorithmen}
}{
	\RequirePackage{cleveref}
	\def\cref{\Cref}
	\def\crefrange{\Crefrange}
		\crefname{figure}{Fig.}{Figs.}
		\Crefname{figure}{Figure}{Figures}
		\crefname{table}{Tab.}{Tabs.}
		\Crefname{table}{Table}{Tables}
		\crefname{chapter}{Chapter}{Chapters}
		\Crefname{chapter}{Chapter}{Chapters}
 	\crefname{section}{Section}{Sections}
		\Crefname{section}{Section}{Sections}
	\crefname{lstlisting}{listing}{listings}
	\Crefname{lstlisting}{Listing}{Listings}
	\crefname{listing}{listing}{listings}
	\Crefname{listing}{Listing}{Listings}
	\crefname{algorithm}{algorithm}{algorithms}
	\Crefname{algorithm}{Algorithm}{Algorithms}
		\crefname{equation}{Eq.}{Eqs.}
		\Crefname{equation}{Equation}{Equations}
}

% Some more language-dependent commands
\ifthenelse{\boolean{germanthesis}}{
	\newcommand{\thedate}{\today}
	\newcommand{\glossarytitlename}{Abkürzungsverzeichnis}
}{
	\newcommand{\thedate}{\selectlanguage{ngerman}\today\selectlanguage{american}}
	\newcommand{\glossarytitlename}{List of Acronyms}
}

% Language-independent timestamp
% \newcommand{\timestamp}{\renewcommand{\dateseparator}{-}\yyyymmdddate\today\quad\settimeformat{hhmmsstime}\currenttime}
\newcommand{\timestamp}{%
	\@dtm@year-\@dtm@month-\@dtm@day\quad\@dtm@hour:\@dtm@minute:\@dtm@second
}

% -----------------------------------------------------------------------------


% -----------------------------------------------------------------------------
% All the simple stuff goes here
% ----------------------------------------------------------------------------- 

%Imports biblatex package
\RequirePackage[backend=biber]{biblatex}

% The Acronym package
\RequirePackage[printonlyused]{acronym}

% The appendix package
\usepackage[toc]{appendix}

% Make automatically inserted "blank pages" really blank
\RequirePackage{emptypage}

% Fix spacing around "..."
\RequirePackage{ellipsis}

% TeX ist kein Zeichenprogramm
\RequirePackage{pgfplotstable} % also loads tikz
\pgfplotsset{compat=newest}% <- set a compat!
\usepgfplotslibrary{fillbetween, groupplots}
\usetikzlibrary{calc}

% Ensure that no floats are typeset before they are referenced
\RequirePackage{flafter}

% Provide the \FloatBarrier command that flushes all pending floats
% \RequirePackage{placeins}
\RequirePackage[section]{placeins}

% Nag about deprecated LaTeX packages and commands
\RequirePackage[l2tabu, orthodox]{nag}

% Define command that will return current filename
\RequirePackage{currfile}

% Do not reset footnote counter on new chapter
\RequirePackage[]{chngcntr}
\counterwithout{footnote}{chapter}

% "Draft" watermark
\newcommand{\thewatermark}{\ifthenelse{\boolean{watermark}}{%
	\begin{tikzpicture}[remember picture, overlay]
		\node[anchor=south, outer sep=3.75cm] at (current page.south) {
			\color{gray}\currfilename\quad\timestamp
		};
	\end{tikzpicture}%
}{}}

% The "algorithm" environment
\RequirePackage{algorithmic}[2009/08/24]
\RequirePackage[plain,chapter]{algorithm}

\renewcommand\fs@plain{%
\def\@fs@cfont{\rmfamily}%
\let\@fs@capt\floatc@plain%
\def\@fs@pre{\hrule\relax\kern2pt}%
\def\@fs@post{}%
\def\@fs@mid{\kern2pt\hrule\relax\vspace\abovecaptionskip\relax}%
\let\@fs@iftopcapt\iffalse%
}
\floatstyle{plain}
\restylefloat{algorithm}
\ifthenelse{\boolean{germanthesis}}{
	\floatname{algorithm}{Algorithmus}
}{
}

% Chapter/section headings and page styles
%\RequirePackage[pagestyles,nobottomtitles*]{titlesec}
\RequirePackage[pagestyles]{titlesec}
\ifthenelse{\boolean{plainunnumbered}}{
\renewpagestyle{plain}{%
	\sethead{}{}{}
	\setfoot{}{}{\thewatermark}
}
}{
\renewpagestyle{plain}{%
	\sethead{}{}{}
	\setfoot{}{\mypageheadfont\thepage}{\thewatermark}
}
}

% Pagestyle "fancy"
\newpagestyle{fancy}{%
\sethead*{
	\bfseries
	\toptitlemarks
	\mypageheadfont\bfseries
	\ifthechapter
		{
			\firsttitlemarks\ifthesection
			{\thesection~\sectiontitle}
			{\thechapter~\chaptertitle}
		}
		{\chaptertitle}
}
{}{\mypageheadfont\bfseries\thepage}
	\setfoot{}{}{\thewatermark}
	\headrule
}
\pagestyle{fancy}

% rename (sub)paragraph to (sub)subsubsubsection
\let\myparagraph\paragraph
\renewcommand{\paragraph}{\FIXME{in order to avoid confusion with hyperref, \\paragraph has been renamed to \\subsubsubsection}}
\let\subsubsubsection\myparagraph
\let\mysubparagraph\subparagraph
\renewcommand{\subparagraph}{\FIXME{in order to avoid confusion with hyperref, \\subparagraph has been renamed to \\subsubsubsubsection}}
\let\subsubsubsubsection\mysubparagraph

% heading styles
\titleformat{\chapter}[display]{\myheadingfont\huge\bfseries\titlerule\vspace{0.25cm}}{\LARGE\chaptertitlename~\thechapter}{1em}{}[\vspace{0.25cm}\titlerule]
\titleformat{\section}{\myheadingfont\Large\bfseries}{\thesection}{1em}{}
\titleformat{\subsection}{\myheadingfont\large\bfseries}{\thesubsection}{1em}{}
\titleformat{\subsubsection}{\myheadingfont\normalsize\bfseries}{\thesubsubsection}{1em}{}
\titleformat{\paragraph}{\myheadingfont\normalsize\bfseries}{\theparagraph}{1em}{}
\titleformat{\subparagraph}{\myheadingfont\normalsize}{\thesubparagraph}{1em}{\labelitemi~}
\titlespacing*{\chapter}{0pt}{50pt}{40pt}
\titlespacing*{\section}{0pt}{3.5ex plus 1ex minus .2ex}{2.3ex plus .2ex}
\titlespacing*{\subsection}{0pt}{3.25ex plus 1ex minus .2ex}{1.5ex plus .2ex}
\titlespacing*{\subsubsection}{0pt}{3.25ex plus 1ex minus .2ex}{1.5ex plus .2ex}
\titlespacing*{\paragraph}{0pt}{3.25ex plus 1ex minus .2ex}{1.5ex plus .2ex}
\titlespacing*{\subparagraph}{0pt}{3.25ex plus 1ex minus .2ex}{1.5ex plus .2ex}
\ifthenelse{\boolean{earlydraft}}{
	\setcounter{tocdepth}{3}
	\setcounter{secnumdepth}{3}
}{
	\setcounter{tocdepth}{3}
	\setcounter{secnumdepth}{3}
}
% -----------------------------------------------------------------------------


% -----------------------------------------------------------------------------
% Commands for setting author, title, ...
% -----------------------------------------------------------------------------
\renewcommand{\author}[1]{\gdef\theauthor{#1}}
\renewcommand{\title}[1]{\gdef\thetitle{#1}}
\newcommand{\germantitle}[1]{\gdef\thegermantitle{#1}}
\newcommand{\thesistype}[1]{\gdef\thethesistype{#1}}
\newcommand{\course}[1]{\gdef\thecourse{#1}}
\newcommand{\academicdegree}[1]{\gdef\theacademicdegree{#1}}
\newcommand{\thesiscite}[1]{\gdef\thethesiscite{#1}}
\newcommand{\birthday}[1]{\gdef\thebirthday{#1}}
\newcommand{\birthplace}[1]{\gdef\thebirthplace{#1}}
\newcommand{\thesisstart}[1]{\gdef\thethesisstart{#1}}
\newcommand{\thesisend}[1]{\gdef\thethesisend{#1}}
\newcommand{\advisors}[1]{\gdef\theadvisors{#1}}
\newcommand{\department}[1]{\gdef\thedepartment{#1}}
\newcommand{\germandepartment}[1]{\gdef\thegermandepartment{#1}}
\newcommand{\chair}[1]{\gdef\thechair{#1}}
\newcommand{\germanchair}[1]{\gdef\thegermanchair{#1}}
\newcommand{\address}[1]{\gdef\theaddress{#1}}
\newcommand{\watermarklogo}[1]{\gdef\thewatermarklogo{#1}}
\newcommand{\chairlogo}[1]{\gdef\thechairlogo{#1}}
\newcommand{\faulogo}[1]{\gdef\thefaulogo{#1}}
\newcommand{\website}[1]{\gdef\thewebsite{#1}}
\newcommand{\fakulty}[1]{\gdef\thefakulty{#1}}
\newcommand{\coverpagepdf}[1]{\gdef\thecoverpagepdf{#1}}
\newcommand{\matriculationnumber}[1]{\gdef\thematriculationnumber{#1}}
% -----------------------------------------------------------------------------


% -----------------------------------------------------------------------------
% Listings
% -----------------------------------------------------------------------------
\RequirePackage[savemem]{listings}
\ifthenelse{\boolean{germanthesis}}{
	\renewcommand{\lstlistlistingname}{Listingverzeichnis}
}{
	\renewcommand{\lstlistlistingname}{List of Listings}
}
\lstset{
	language=C,
	frame=lines,
	framesep=5pt,
	captionpos=b,
	abovecaptionskip=1em,
	numbers=left,
	xleftmargin=15pt,
	framexleftmargin=15pt,
	numberstyle=\tiny,
	numbersep=5pt,
	stepnumber=1,
	fontadjust,
	tabsize=2,
	showtabs=false,
	showspaces=false,
	showstringspaces=false,
	breaklines=true,
	breakatwhitespace=true,
	keywordstyle=\color{black},
	identifierstyle=\bfseries\color{blue},
	commentstyle=\color{red},
	prebreak=\raisebox{0ex}[0ex][0ex]{\color{gray}\ensuremath{\searrow}},
	stringstyle=\bfseries\color{blue},
	basicstyle=\ttfamily\bfseries\color{black}\footnotesize
}
\lstdefinestyle{txt}{
	float,
	language=C
}
\lstdefinestyle{xml}{
	float,
	language=XML
}


% -----------------------------------------------------------------------------


% -----------------------------------------------------------------------------
% TOCs
% -----------------------------------------------------------------------------

% Include list-of-Xs in table of contents
\ifthenelse{\boolean{earlydraft}}{
	\RequirePackage[]{tocbibind}
	\renewcommand{\lstlistoflistings}{\begingroup\tocfile{\lstlistlistingname}{lol}\endgroup}
	\newcommand{\mymakeglossarytocentry}{\addcontentsline{toc}{chapter}{\glossarytitlename}}
	\newcommand{\mymaketodotocentry}{\todototoc}
}{
	\RequirePackage[nottoc,notlot,notlof]{tocbibind}
	\newcommand{\mymakeglossarytocentry}{}
	\newcommand{\mymaketodotocentry}{}
}
\RequirePackage{titletoc}

% -----------------------------------------------------------------------------


% -----------------------------------------------------------------------------
% Visible TODO and FIXME markers
% -----------------------------------------------------------------------------
\ifthenelse{\boolean{germanthesis}}{
	\RequirePackage[colorinlistoftodos, textwidth=3.75cm, ngerman]{todonotes}
	\@todonotes@SetTodoListName{Offene Punkte}
}{
	\RequirePackage[colorinlistoftodos, textwidth=3.75cm]{todonotes}
}
\newcounter{TODOCOUNT}
\newcommand{\TODO}[1]{\vspace{0.5em}\todo[inline, color=orange]{#1}\stepcounter{TODOCOUNT}}
\newcommand{\FIXME}[1]{\todo[size=\small, color=red]{#1}\stepcounter{TODOCOUNT}}
\AtEndDocument{
	\ifnum\value{TODOCOUNT}>0
		%\cleardoublepage
		\mymaketodotocentry\listoftodos
	\fi
}
% -----------------------------------------------------------------------------


% -----------------------------------------------------------------------------
% Title page
% -----------------------------------------------------------------------------
\RequirePackage{pdfpages}% to easier change the cover page using a pdf
\newcommand{\coverpagetextsmall}[1]{%
	\@groupcoverpage@smalltextsize%
	#1%
	\vspace{0.5cm}\\%
}

\newcommand{\coverpagetextlarge}[1]{%
	\vspace{0.25cm}%
	\@groupcoverpage@largetextsize%
	#1%
	\@groupcoverpage@smalltextsize%
	\vspace{0.5cm}\\%
}
\renewcommand{\maketitle}{%

% Do some stuff we had to postpone until after e.g. \thetitle was set
\hypersetup{%
  pdftitle = {\thetitle},
  pdfsubject = {\thethesistype{} im Fach \thecourse},
  pdfauthor = {\theauthor},
  pdfborder=0 0 0
}

\pagenumbering{alph}

% Output cover page
{%

\tikzoption{text @groupcoverpage@ badly raggedleft}[]{\def\tikz@text@action{\raggedleft\relax}}

\begingroup
\pagestyle{empty}
\setcounter{page}\@ne

\def\@groupcoverpage@titlesize{\@setfontsize{\@groupcoverpage@titlesize}{20}{24pt}}
\def\@groupcoverpage@subtitlesize{\@setfontsize{\@groupcoverpage@subtitlesize}{14}{17pt}}
\def\@groupcoverpage@citeassize{\@setfontsize{\@groupcoverpage@citeassize}{8}{9pt}}
\def\@groupcoverpage@addressize{\@setfontsize{\@groupcoverpage@addressize}{8}{9pt}}
\def\@groupcoverpage@smalltextsize{\@setfontsize{\@groupcoverpage@smalltextsize}{14}{17pt}}
\def\@groupcoverpage@largetextsize{\@setfontsize{\@groupcoverpage@largetextsize}{20}{24pt}}



\renewcommand{\baselinestretch}{1.00}
\ifthenelse{\boolean{cscoversheet}}{
\begin{tikzpicture}[remember picture,overlay] \node[xshift=0cm,yshift=0cm] at (current page.south west) { \begin{tikzpicture}[remember picture, overlay]
	\definecolor{faublue}{rgb}{0.0,0.2,0.4}

	%%% logos
	\ifthenelse{\boolean{earlydraft}}{
		% no watermark in [earlydraft] mode
	}{
	\AddToHookNext{shipout/background}{%
		\put (0in,-25cm){\includegraphics[width=\paperwidth]{\thewatermarklogo}}%
	}
	}

	\pgftext[top,left,at={\pgfpoint{2.5cm}{27cm}}] {%
		\includegraphics[width=7cm]{\thefaulogo}%
	}

	\pgftext[top,left,at={\pgfpoint{2.5cm}{4cm}}] {%
		\includegraphics[height=2cm]{\thechairlogo}%
	}

	%%% single-line text
	\pgftext[base,left,at={\pgfpoint{2.3cm}{22cm}}] {%
		\raggedright%
		\sffamily%
		\bfseries%
		\@groupcoverpage@titlesize%
		\renewcommand{\baselinestretch}{1.00}%
		\ifthenelse{\boolean{germanthesis}}{
			\thegermandepartment%
		}{
			\thedepartment%
		}
	}
	\pgftext[base,left,at={\pgfpoint{2.37cm}{21cm}}] {%
		\raggedright%
		\sffamily%
		\bfseries%
		\@groupcoverpage@subtitlesize%
		\renewcommand{\baselinestretch}{1.00}%
		\ifthenelse{\boolean{germanthesis}}{
			\thegermanchair%
		}{
			\thechair%
		}
	}

	%%% multi-line text
	\pgftext[top,left,at={\pgfpoint{2.5cm}{18cm}}] {%
		\parbox{15cm}{%
			\raggedright%
			\sffamily%
			\@groupcoverpage@largetextsize%
			\renewcommand{\baselinestretch}{1.25}%
			\coverpagetextsmall{\nohyphens{\theauthor}}%
			\coverpagetextlarge{\nohyphens{\thetitle}}%
			\coverpagetextsmall{\thethesistype{} im Fach \thecourse}%
			\coverpagetextsmall{\thethesisend}%
		}%
	};
	\draw[xshift=2.5cm,yshift=6cm] node [above right,text width=15cm, text badly ragged] {%
		\sffamily% 
		\@groupcoverpage@citeassize%
		\renewcommand{\baselinestretch}{1.00}%
		\ifthenelse{\boolean{germanthesis}}{%
			Bitte zitieren als:\\%
			\nohyphens{\theauthor, ``\thetitle,'' \thethesiscite, Friedrich-Alexander-Universität Erlangen-Nürnberg, \thegermandepartment, \thegermanchair, \DTMenglishmonthname{\@dtm@month}~\@dtm@year.}%
		}{%
			\begin{otherlanguage*}{english}%
				Please cite as:\\%
				\nohyphens{\theauthor, ``\thetitle,'' \thethesiscite, Friedrich-Alexander-Universität Erlangen-Nürnberg, \thedepartment, \thechair, \DTMenglishmonthname{\@dtm@month}~\@dtm@year.}%
			\end{otherlanguage*}
		}
	};
	\draw[color=faublue, line width=0.8pt] (1.25cm,23.6cm) -- (16.7cm,23.6cm) -- (16.7cm,22.6cm) -- (17.7cm,22.6cm) -- (17.7cm,23.6cm) -- (19.75cm,23.6cm);
	\draw[color=faublue, line width=0.8pt] (1.25cm,19.6cm) -- (19.75cm,19.6cm);
	\draw[color=faublue, line width=0.8pt] (1.25cm,4.6cm) -- (19.75cm,4.6cm);

	\draw[xshift=19.08cm,yshift=4.05cm] node [below left,text width=12cm, text @groupcoverpage@ badly raggedleft] {%
		\sffamily%
		\@groupcoverpage@addressize%
		\renewcommand{\baselinestretch}{1.00}%
		Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)\\%
		\ifthenelse{\boolean{germanthesis}}{
			\thegermandepartment\\%
			\thegermanchair\\%
		}{
			\thedepartment\\%
			\thechair\\%
		}
		\vspace{0.5em}%
		\theaddress%
	};
	\draw[xshift=19.08cm,yshift=1.8cm] node [above left,text width=12cm, text @groupcoverpage@ badly raggedleft] {%
		\sffamily%
		\@groupcoverpage@addressize%
		\renewcommand{\baselinestretch}{1.00}%
		\href{\thewebsite}{\thewebsite}%
	};
\end{tikzpicture} }; \end{tikzpicture}

\cleardoublepage
\setcounter{page}\@ne
}{
}
\endgroup
}

% Title pages use single line spacing, no parindent and no parskips
\begin{singlespace}
\setlength\parindent{0pt}
\setlength\parskip{0pt}

%
% Now output the title page
%

\pagestyle{empty}
\pagenumbering{roman}

\ifthenelse{\boolean{phdthesis}}
{
\thispagestyle{empty}
{
	% Adopted from FAU style guides: https://www.intern.fau.de/kommunikation-und-marke/vorlagen/abschlussarbeiten/#tf
	% CC0 1.0 licensed by Stefan Wolff, 2025
	\newgeometry{top=12mm, bottom=0mm, left=25mm, right=12mm}
	\raggedright
	\begingroup
	\tikzset{every picture/.style={}}

	\ifthenelse{\boolean{colorcoversheet}}{
		\pagecolor[RGB]{4,49,106}
		\begin{tikzpicture}
		\node[anchor=north west, inner sep=0] at (0,\textheight) {\includegraphics[trim={17.1mm 0 17.1mm 17.1mm}, clip, width=66.8mm]{i7/FAU_Wortmarke_white.pdf}};
		\ifthenelse{\boolean{germanthesis}}{
			\node[anchor=north east, inner sep=0] at (\textwidth,\textheight) {\includegraphics[trim={17.1mm 0 17.1mm 16.5mm}, clip, width=66.8mm]{i7/FAU_UK_Techfak_white.pdf}};
		}{
			\node[anchor=north east, inner sep=0] at (\textwidth,\textheight) {\includegraphics[trim={17.1mm 0 17.1mm 16.5mm}, clip, width=66.8mm]{i7/FAU_UK_Techfak_EN_white.pdf}};
		}
		\node[anchor=south east, inner sep=0, xshift=-13.6mm, yshift=10mm] at (\textwidth,0) {\includegraphics[trim={21.74mm 21.74mm 21.74mm 21.74mm}, clip, width=40mm]{i7/FAU_Siegel_white.pdf}};
		\node[color=white, right, inner sep=-2pt] at (0,200mm) {\fontsize{40}{48}\selectfont\bfseries\sffamily \thethesistype{}\hfill\par};
		\draw[color=white, line width=1.6pt] (0,180mm) -- (\textwidth,180mm);
		\node[color=white, right, text width=\textwidth, inner sep=0] at (0,160mm) {\fontsize{17}{20.4}\selectfont\sffamily\thethesisend\hfill\par};
		\draw[color=white, line width=1.6pt] (0,140mm) -- (\textwidth,140mm);
		\node[color=white, right, text width=\textwidth, inner sep=0] at (0,120mm) {\fontsize{24}{28.8}\selectfont\sffamily\theauthor\hfill\par};
		\draw[color=white, line width=1.6pt] (0,100mm) -- (\textwidth,100mm);
		\node[color=white, right, text width=\textwidth, inner sep=0] at (0,80mm) {\fontsize{17}{20.4}\selectfont\slshape\bfseries\sffamily\thetitle\hfill\par};
		\draw[color=white, line width=1.6pt] (0,60mm) -- (\textwidth,60mm);
		\end{tikzpicture}
	}{% else
		\begin{tikzpicture}
		\node[anchor=north west, inner sep=0] at (0,\textheight) {\includegraphics[trim={17.1mm 0 17.1mm 17.1mm}, clip, width=66.8mm]{i7/FAU_Wortmarke_black.pdf}};
		\ifthenelse{\boolean{germanthesis}}{
			\node[anchor=north east, inner sep=0] at (\textwidth,\textheight) {\includegraphics[trim={17.1mm 0 17.1mm 16.5mm}, clip, width=66.8mm]{i7/FAU_UK_Techfak_black.pdf}};
		}{
			\node[anchor=north east, inner sep=0] at (\textwidth,\textheight) {\includegraphics[trim={17.1mm 0 17.1mm 16.5mm}, clip, width=66.8mm]{i7/FAU_UK_Techfak_EN_black.pdf}};
		}
		\node[anchor=south east, inner sep=0, xshift=-13.6mm, yshift=10mm] at (\textwidth,0) {\includegraphics[trim={21.74mm 21.74mm 21.74mm 21.74mm}, clip, width=40mm]{i7/FAU_Siegel_black.pdf}};
		\node[color=black, right, inner sep=-2pt] at (0,200mm) {\fontsize{40}{48}\selectfont\bfseries\sffamily \thethesistype{}\hfill\par};
		\draw[color=black, line width=1.6pt] (0,180mm) -- (\textwidth,180mm);
		\node[color=black, right, text width=\textwidth, inner sep=0] at (0,160mm) {\fontsize{17}{20.4}\selectfont\sffamily\thethesisend\hfill\par};
		\draw[color=black, line width=1.6pt] (0,140mm) -- (\textwidth,140mm);
		\node[color=black, right, text width=\textwidth, inner sep=0] at (0,120mm) {\fontsize{24}{28.8}\selectfont\sffamily\theauthor\hfill\par};
		\draw[color=black, line width=1.6pt] (0,100mm) -- (\textwidth,100mm);
		\node[color=black, right, text width=\textwidth, inner sep=0] at (0,80mm) {\fontsize{17}{20.4}\selectfont\slshape\bfseries\sffamily\thetitle\hfill\par};
		\draw[color=black, line width=1.6pt] (0,60mm) -- (\textwidth,60mm);
		\end{tikzpicture}
	} % fi
	\endgroup
	\restoregeometry
}

\nopagecolor
\cleardoublepage
\thispagestyle{empty}

\vspace*{1cm}

\begin{center}
\begin{singlespace}
\LARGE
\textbf{\thetitle}\\
\vspace*{.5\baselineskip}
---\\
\vspace*{.5\baselineskip}
\textbf{\thegermantitle}\\
\end{singlespace}

\vspace*{2.5cm}%{3cm}

\begin{doublespacing}
\large
Der \thefakulty~Fakultät\\
der Friedrich-Alexander-Universität\\
Erlangen-Nürnberg\\
zur\\
Erlangung des Doktorgrades\\[3em]

\textsc{\textls[200]{\theacademicdegree}}\\[3em]
vorgelegt von\\
\theauthor\\
aus \thebirthplace\\
\end{doublespacing}
\end{center}

\clearpage

\thispagestyle{empty}

\vspace*{14cm}


\begin{onehalfspacing}
Als Dissertation genehmigt\\
von der \thefakulty~Fakultät\\
der Friedrich-Alexander-Universität Erlangen-Nürnberg\\[2em]
\begin{tabular}{@{}lp{6cm}}
	Tag der mündlichen Prüfung:& \textbf{\thethesisend}\\[2em]

Gutachter: & \textbf{\theadvisors}\\
\end{tabular}
\end{onehalfspacing}


\cleardoublepage

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% third page
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%\newpage
%\thispagestyle{empty}

%\begin{center}
%{\em To my father,\\
%who taught me some of the most important things about life\\
%but, unfortunately, died much too early.\\
%(1938-1998)
%}
%\end{center}

%\vspace*{12cm}
%{\em 
%\hfill “Freedom is the right to tell people

%\hfill what they do not want to hear.”

%\hfill -- George Orwell\\
%}

}
% else: Master Thesis title page
{
\thispagestyle{empty}

\vspace*{0.8cm}

\begin{center}

{\LARGE \textbf{\thetitle}}\\[2\baselineskip]

{\large
\thethesistype{} im Fach \thecourse\\[2\baselineskip]

vorgelegt von\\[2\baselineskip]

\textbf{\theauthor}\\[1\baselineskip]

geb. am \thebirthday\\
in \thebirthplace\\[2\baselineskip]

angefertigt am\\[2\baselineskip]

\textbf{
\thegermanchair\\[1.2\baselineskip]
\thegermandepartment\\[0.2\baselineskip]
Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)\\[0.2\baselineskip]
}
}~\\[1\baselineskip]

\begin{tabular}{>{\raggedleft}p{5cm}p{6cm}}
Betreuer:	& \textbf{\theadvisors}\\
~ & ~\\
Abgabe der Arbeit: & \textbf{\thethesisend}\\
\end{tabular}

\end{center}

\clearpage
}
\end{singlespace}

%
% The delaration comes here
%

\ifthenelse{\boolean{phdthesis}}{
%the dissertation does not need this part
}
{
\thispagestyle{empty}

\begin{otherlanguage*}{ngerman}
{\Large\noindent%
Eigenständigkeitserklärung\newline}
\noindent%
Hiermit versichere ich, \theauthor{} (Matrikelnummer~\thematriculationnumber{}), die vorgelegte Arbeit selbstständig und ohne unzulässige Hilfe Dritter sowie ohne die Hinzuziehung nicht offengelegter und insbesondere nicht zugelassener Hilfsmittel angefertigt zu haben. Die Arbeit hat in gleicher oder ähnlicher Form noch keiner anderen Prüfungsbehörde vorgelegen und wurde auch von keiner anderen Prüfungsbehörde bereits als Teil einer Prüfung angenommen. \\
Die Stellen der Arbeit, die anderen Quellen im Wortlaut oder dem Sinn nach entnommen wurden, sind durch Angaben der Herkunft kenntlich gemacht. Dies gilt auch für Zeichnungen, Skizzen, bildliche Darstellungen sowie für Quellen aus dem Internet. \\
Mir ist insbesondere bewusst, dass die Nutzung künstlicher Intelligenz verboten ist, sofern diese nicht ausdrücklich als Hilfsmittel von dem Prüfer bzw. der Prüferin zugelassen wurde. Dies gilt insbesondere für Chatbots (bspw. ChatGPT) bzw. allgemein solche Programme, die anstelle meiner Person die Aufgabenstellung der Prüfung bzw. Teile derselben bearbeiten könnten. \\
Verstöße gegen die o.g. Regeln sind als Täuschung bzw. Täuschungsversuch zu qualifizieren und führen zu einer Bewertung der Arbeit mit „nicht bestanden“. \\
\end{otherlanguage*}%
\newline
\begin{otherlanguage*}{american}
{\Large\noindent%
Declaration of Originality\newline}
\noindent%
I, \theauthor{} (student registration number~\thematriculationnumber{}), hereby confirm that I completed the submitted work independently and without the unauthorized assistance of third parties and without the use of undisclosed and, in particular, unauthorized aids. This work has not been previously submitted in its current form or in a similar form to any other examination authorities and has not been accepted as part of an examination by any other examination authority. \\
Where the wording has been taken from other people’s work or ideas, this has been properly acknowledged and referenced. This also applies to drawings, sketches, diagrams and sources from the Internet. \\
In particular, I am aware that the use of artificial intelligence is forbidden unless its use an aid has been expressly permitted by the examiner. This applies in particular to chatbots (e.g., ChatGPT) and such programs in general that can complete the tasks of the examination or parts thereof on my behalf. \\
Any infringements of the above rules constitute fraud or attempted fraud and shall lead to the examination being graded “fail” (“nicht bestanden”).
\end{otherlanguage*}


\vspace{1.5cm}


\noindent%
(\theauthor)\\
Erlangen, \thedate

\cleardoublepage
}

\pagestyle{fancy}

}
% -----------------------------------------------------------------------------
