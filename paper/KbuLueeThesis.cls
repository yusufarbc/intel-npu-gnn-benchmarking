\NeedsTeXFormat{LaTeX2e} %% 07.02.2025 Ulvi Kanbur
\ProvidesClass{KbuLueeThesis}
\PassOptionsToPackage{fleqn}{amsmath}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{scrreprt}}
\ProcessOptions
\LoadClass[12pt,%
parskip=full,%
listof=entryprefix,%
chapterprefix=on,%
toc=chapterentrydotfill,%
numbers=endperiod%
]%
{scrreprt}
\RequirePackage[bottom=25mm,top=30mm,left=40mm,right=25mm]{geometry}
\RequirePackage[onehalfspacing]{setspace} % Set space between lines
\linespread{1.425} % for one half spacing match with MS Word (tested on Office 365)
\setlength{\footskip}{34pt} %% for vertical page number position
\RequirePackage{enumitem} % to adjust itemize/enumerate environments
\setlist[1]{leftmargin=1cm} % set the first level item indentation to 1 cm for both itemize and enumerate environments
\RequirePackage[british]{babel}  %recommended by biblatex, for languages other than American English
\RequirePackage[T1]{fontenc} % Standard package for selecting font encodings
%\RequirePackage{newtx}  % Alternative uses of the TX fonts, with improved metrics, seem more recent
\RequirePackage{newtxtext}  % for Times New Roman
\RequirePackage{newtxmath} % for Times New Roman mathematical symbols
\RequirePackage{amsmath} %American Mathematical Society mathematical facilities https://ctan.org/pkg/amsmath
\RequirePackage{graphicx}   % Required for inserting images
% -- enforce zero indent AFTER all packages
\AtBeginDocument{%
  \makeatletter
  \@fleqntrue
  \setlength{\mathindent}{2pt}% or 2em if you want a small left margin
  \makeatother
}
\RequirePackage{subcaption}
\RequirePackage{mfirstuc} % Uppercase the first letter of a word https://ctan.org/pkg/mfirstuc
\RequirePackage[final,spacing=true,kerning=true]{microtype}
%\microtypecontext{spacing=nonfrench} % overleaf suggested that 
\RequirePackage{calc}
\RequirePackage{float}
\RequirePackage{array}
\RequirePackage{tabularx}
%\RequirePackage{longtable} 
\RequirePackage{makecell}
\RequirePackage{multirow}
\RequirePackage{booktabs} 
\RequirePackage{xltabular} 
\RequirePackage{mwe} % provides example-image.png (no external file needed)
%%%REF STYLE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage{csquotes}%recommended by biblatex
\RequirePackage{xpatch} %recommended by biblatex
\RequirePackage[backend=bibtex,style=ieee,doi=false,url=false,isbn=false,related=false,citestyle=numeric-comp]{biblatex}

% Remove common optional fields before printing
\AtEveryBibitem{%
  \clearfield{note}%
%  \clearfield{url}%
%  \clearfield{doi}%
  \clearfield{issn}%
%  \clearfield{isbn}%
  \clearfield{eprint}%
  \clearfield{urldate}%
  \clearfield{month}%
  \clearfield{day}%
%  \clearfield{edition}%
  \clearfield{series}%
  \clearfield{eventtitle}%
  \clearlist{language}%
}


% \DeclareFieldFormat{journaltitle}{\textbf{\mkbibemph{#1}}}     % Journal names
% %\DeclareFieldFormat{booktitle}{\textbf{\mkbibemph{#1}}}        % Book/proceedings titles
% \DeclareFieldFormat{maintitle}{\textbf{\mkbibemph{#1}}}        % Multi-volume work
% \DeclareFieldFormat{issuetitle}{\textbf{\mkbibemph{#1}}}       % Issue title
% %\DeclareFieldFormat{series}{\textbf{\mkbibemph{#1}}}           % Series name
% \DeclareFieldFormat{eventtitle}{\textbf{\mkbibemph{#1}}}       % Conference/event
% \DeclareFieldFormat{organization}{\textbf{\mkbibemph{#1}}}     % Organization
% \DeclareFieldFormat{institution}{\textbf{\mkbibemph{#1}}}      % Institution (thesis/report)
% \DeclareListFormat{publisher}{\textbf{\mkbibemph{#1}}}

\DeclareFieldFormat[article,book,inbook,collection,incollection,proceedings,inproceedings]{volume}{#1}%
\DeclareFieldFormat[article,book,inbook,collection,incollection,proceedings,inproceedings]{number}{#1}%
\DeclareFieldFormat[article,book,inbook,collection,incollection,proceedings,inproceedings]{pages}{#1}%
\DeclareFieldFormat[book,inbook,incollection,collection,inproceedings,proceedings]{title}{\mkbibquote{#1}}
\DeclareQuotePunctuation{.}

% % --- Print volume(issue) only here (no labels, no eid here) ---
 \renewbibmacro*{volume+number+eid}{%
   \printfield{volume}%
   \printfield[parens]{number}%
 }

% % --- colon + pages or (if missing) eid ---
% \renewbibmacro*{pages}{%
%   \setunit{\addcolon\space}% ensures one colon before the block
%   \iffieldundef{pages}
%     {\printfield{eid}}%
%     {\printfield{pages}}%
%     \setunit{\addspace}%
%  \iffieldundef{year}{\printdate[year]}{\printtext[parens]{\printfield{year}}}
% }


% % --- Don’t let IEEE print the date elsewhere (no duplicates) ---
% \renewbibmacro*{issue+date}{}
% \renewbibmacro*{date+extrayear}{}

\addbibresource{thesis.bib}
\addbibresource{references.bib}

%%%%%%%%%%%%%%%%%% END REF STYLE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%




\newcommand{\ba}[1]{\textbf{\sffamily #1}} %% for Bold Arial

\newcommand*{\thesisTitle}[1]{\def\theTitle{#1}}
\newcommand*{\thesisTitleTR}[1]{\def\theTitleTR{#1}}
\newcommand*{\thesisDefenseDay}[1]{\def\theDefenseDay{#1}}
\newcommand*{\thesisDegree}[1]{\def\theDegree{#1}}
\newcommand*{\thesisDegreeTR}[1]{\def\theDegreeTR{#1}}
\newcommand*{\thesisProgram}[1]{\def\theProgram{#1}}
\newcommand*{\thesisProgramTR}[1]{\def\theProgramTR{#1}}
\newcommand*{\thesisAuthor}[1]{\def\theAuthor{#1 }}
\newcommand*{\thesisAdvisor}[1]{\def\theAdvisor{#1}}
\newcommand*{\thesisCoAdvisor}[1]{\def\theCoAdvisor{#1}}
\newcommand*{\thesisCoAdvisorInstitution}[1]{\def\theCoAdvisorInstitution{#1}}
\newcommand*{\thesisAdvisorTitle}[1]{\def\theAdvisorTitle{#1}}
\newcommand*{\thesisAdvisorTitleTR}[1]{\def\theAdvisorTitleTR{#1}}
\newcommand*{\thesisAdvisorDepartment}[1]{\def\theAdvisorDepartment{#1}}
\newcommand*{\thesisChairman}[1]{\def\theChairman{#1}}
\newcommand*{\thesisMemberTWO}[1]{\def\theMemberTWO{#1}}
\newcommand*{\thesisMemberTHREE}[1]{\def\theMemberTHREE{#1}}
\newcommand*{\thesisMemberFOUR}[1]{\def\theMemberFOUR{#1}}
\newcommand*{\thesisMemberFIVE}[1]{\def\theMemberFIVE{#1}}
\newcommand*{\thesisMemberSIX}[1]{\def\theMemberSIX{#1}}
\newcommand*{\thesisMemberSEVEN}[1]{\def\theMemberSEVEN{#1}}
\newcommand*{\thesisMember}[2][]{\parbox{0.14\linewidth}{Member} : \parbox{0.65\linewidth}{#2}\parbox{0.2\linewidth}{\dotfill}}

% ------------------------------
% Course-project customization (defaults keep thesis behavior)
% ------------------------------
\newcommand*{\thesisDocTypeEN}[1]{\def\theDocTypeEN{#1}}
\newcommand*{\thesisDocTypeTR}[1]{\def\theDocTypeTR{#1}}
\newcommand*{\thesisAdvisorLabelEN}[1]{\def\theAdvisorLabelEN{#1}}
\newcommand*{\thesisAdvisorLabelTR}[1]{\def\theAdvisorLabelTR{#1}}
\newcommand*{\thesisPreparedAsEN}[1]{\def\thePreparedAsEN{#1}}
\newcommand*{\thesisPreparedAsTR}[1]{\def\thePreparedAsTR{#1}}
\newcommand*{\thesisCourseNameEN}[1]{\def\theCourseNameEN{#1}}
\newcommand*{\thesisCourseNameTR}[1]{\def\theCourseNameTR{#1}}

\def\theDocTypeEN{\theDegree\ THESIS}
\def\theDocTypeTR{\theDegreeTR\ Tezi}
\def\theAdvisorLabelEN{Thesis Advisor}
\def\theAdvisorLabelTR{Tez Danışmanı}
\def\thePreparedAsEN{PhD Thesis}
\def\thePreparedAsTR{Doktora Tezi}
\def\theCourseNameEN{}
\def\theCourseNameTR{}

\newcommand*{\thesisMonth}{\ifcase \month \or January\or February\or March\or %
April\or May \or June\or July\or August\or September\or October\or November\or %
December\fi} 
\newcommand*{\thesisMonthTR}{\ifcase \month \or Ocak\or Şubat\or Mart\or %
Nisan\or Mayıs \or Haziran\or Temmuz\or Ağustos\or Eylül\or Ekim\or Kasım\or %
Aralık\fi}
\def\dday{\ifthenelse{\theDefenseDay<10}{0}{}\theDefenseDay}
\def\dmonth{\ifthenelse{\number\month<10}{0}{}\number\month}


\newcommand{\thesisAbstract}[1]{\def\theAbstract{#1}}
\newcommand{\thesisOzet}[1]{\def\theOzet{#1}}
\newcommand*{\thesisPages}[1]{\def\thePages{#1}}
\newcommand*{\thesisKeywords}[1]{\def\theKeywords{#1}}
\newcommand*{\thesisKeywordsTR}[1]{\def\theKeywordsTR{#1}}
\newcommand*{\thesisScienceCode}[1]{\def\theScienceCode{#1}}
\newcommand{\thesisAcknowledgment}[1]{\def\theAcknowledgment{#1}}

\newcommand*{\thesisResume}[1]{\def\theResume{#1}}
%\newcommand*{\thesisRefs}[1]{\def\theRefs{#1}}


\newcommand*{\thesisSymbol}[2]{#1 \> : \>\ #2\\}
\newcommand*{\thesisAbbr}[2]{\parbox{0.13\linewidth}{#1} \> : \>\ \parbox[t]{0.87\linewidth}{#2}\\}
\newcommand{\thesisSymbolList}[1]{\def\theGetSymbols{#1}}
\newcommand{\thesisAbbrList}[1]{\def\theGetAbbr{#1}}




\renewcaptionname{british}{\contentsname}{\normalfont\normalsize\centering{\textbf{CONTENTS\\\hfill Page}}}
\renewcaptionname{british}{\listfigurename}{\normalfont\normalsize\centering{\textbf{LIST OF FIGURES\\\hfill Page}}}
\renewcaptionname{british}{\listtablename}{\normalfont\normalsize\centering{\textbf{LIST OF TABLES\\\hfill Page}}}
\def\@dotsep{0.072} %% dot separation in TOC, LOF, LOT
\BeforeTOCHead{%
  \KOMAoptions{parskip=false}% no parskip in ToC
  \RedeclareSectionCommand[beforeskip=20mm, afterskip=0mm]{chapter}% 20 mm before after ToC title
}


\renewcommand{\@chapapp}{PART} % write PART instead of Chapter
\renewcommand{\raggedchapter}{\centering}
\addtokomafont{chapterentry}{\normalfont}
\setkomafont{disposition}{\bfseries}

\renewcommand*{\chapterformat}{\mbox{\chapappifchapterprefix{\nobreakspace}\thechapter\IfUsePrefixLine{}{\enskip}}}% no point after PART number, adapted from KOMA-script guide with date 2023-06-16

\RedeclareSectionCommand[%
    style=chapter,%
    beforeskip=-20mm,%
    innerskip=0pt,%
    tocbeforeskip=0.5\baselineskip,%
    afterskip=0pt,%
    tocindent=0pt,% 
    tocnumwidth=5em,%
    font=\normalsize%
]{chapter}

\DeclareNewSectionCommand[%
    style=chapter,%
    beforeskip=-20mm,%
    level=0,%
    toclevel=0,%    
    innerskip=0pt,%
    afterskip=0pt,%
    tocindent=0pt,% 
    tocnumwidth=5em,%
    font=\normalsize%
]{thesisChapter}

\RedeclareSectionCommands[%
font=\normalsize,%
tocindent=5mm,%
tocnumwidth=9.5mm%
]{section}

\RedeclareSectionCommands[%
font=\normalsize,%
tocindent=10mm,%
tocnumwidth=13mm%
]{subsection}

\RedeclareSectionCommands[%
font=\normalsize,%
tocindent=15mm,%
tocnumwidth=16mm,%
level=2,%
toclevel=2%
]{subsubsection}


%\setcapindent{0mm}
\renewcommand*{\captionformat}{~} % caption of figures and tables, remove colon symbol :

%\newcommand*{\adddot}[1]{#1\unskip.\hfil}
%\DeclareTOCStyleEntries[breakafternumber=on,entrynumberformat=\chapapp\ ,pagenumberbox=\mbox,rightindent=0em]{tocline}{chapter,thesisChapter}

\DeclareTOCStyleEntries[breakafternumber=on,entrynumberformat=\def\autodot{}PART\ ,pagenumberbox=\mbox,rightindent=0em]{tocline}{chapter,thesisChapter}

\DeclareTOCStyleEntries[pagenumberbox=\mbox,rightindent=0em]{tocline}{section,subsection,subsubsection,figure,table}



%%----------- START of THESIS PREAMBLE ------------------------%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\makeThesisPreamble}{%
%%%--------OUTER COVER---------%%%%
\newgeometry{top=4cm,left=4cm,right=4cm,bottom=5cm}
\thispagestyle{empty}
\begin{singlespace}
\centering\bfseries\Large
\includegraphics[height=3.6cm,width=4.8cm]{logo.png}\\[\baselineskip]
\parbox[b][7\baselineskip][t]{\linewidth}{\centering\ba{\theTitle}}
\ba{\the\year}\\
\ba{\theDocTypeEN}\\
\ba{\MakeUppercase{\theProgram}}\\[4\baselineskip]
\begingroup\makeatletter\ifx\theCourseNameEN\@empty\else\ba{\theCourseNameEN}\\[\baselineskip]\fi\endgroup
\ba{\theAuthor}\\[4\baselineskip]
\ba{\theAdvisorLabelEN}\\*
\ba{\theAdvisorTitle\ \theAdvisor}
\end{singlespace}
%\clearpage
\newpage%

%%%------INNER COVER--------%%%%
\thispagestyle{empty}
\pagenumbering{roman}
\newpage\restoregeometry
\newgeometry{top=5cm,left=4cm,right=4cm,bottom=5cm}
{
\centering\bfseries
\begin{singlespace}
\parbox[t][9\baselineskip][t]{\linewidth}{\centering\theTitle}
\end{singlespace}
\theAuthor\\[3\baselineskip]
\theAdvisorLabelEN\\
\theAdvisorTitle\ \theAdvisor\\[3\baselineskip] 
T.C.\\
Karabük University\\
Institute of Graduate Programs\\
Department of \makefirstuc{\MakeLowercase{\theProgram}}\\
Prepared as\\
\thePreparedAsEN\\[3\baselineskip]
KARABÜK\\
\thesisMonth\ \the\year

}
%%%----APPROVAL PAGE-----%%%
\newpage\restoregeometry
\addxcontentsline{toc}{thesisChapter}{APPROVAL}
\parbox[t][5\baselineskip][t]{\linewidth}{I certify that in my opinion the report submitted by \theAuthor titled  "\theTitle" is fully adequate in scope and in quality as a course project report.}\\
\theAdvisorTitle\ \theAdvisor\hfill\dotfill\\
\theAdvisorLabelEN, Department of \theAdvisorDepartment\\[2\baselineskip]
%\theCoAdvisor, \theCoAdvisorInstitution\hfill\dotfill\\
%Thesis Co-Advisor
This report is accepted by the examining committee with a unanimous vote in the Department of \makefirstuc{\MakeLowercase{\theProgram}} as a course project. \dday .\dmonth .\the\year\\[1\baselineskip]
\underline{Examining Committee Members (Institutions)}\hfill\underline{Signature}\hfill

\parbox{0.14\linewidth}{Chairman} : \parbox{0.65\linewidth}{\theChairman}\parbox{0.2\linewidth}{\dotfill}

\thesisMember{\theMemberTWO}

\thesisMember{\theMemberTHREE}

\thesisMember{\theMemberFOUR}

\thesisMember{\theMemberFIVE}\\[5\baselineskip]
%\thesisMember{\theMemberSIX}
%%%% GIVE A NEW LINE SPACE 
%\thesisMember{\theMemberSEVEN}\\[\baselineskip] %%OPEN THIS and CLOSE ABOVE \\[5\base..]
The degree of PhD by the thesis submitted is approved by the Administrative Board of the Institute of Graduate Programs, Karabük University.\\[1\baselineskip]
Prof. Dr. Zeynep ÖZCAN\hfill\dotfill\\
Director of the Institute of Graduate Programs

%%------DECLARATION-----%%%
\newpage
\begin{singlespace}
\ \vfill\textit{"I declare that all the information within this report has been gathered and presented in accordance with academic regulations and ethical principles and I have according to the requirements of these regulations and principles cited all those which do not originate in this work as well."}
\end{singlespace}

\hfill\theAuthor%
%%%%%----------ABSTRACT---------%%%%%%%%%
\newpage%
\addxcontentsline{toc}{thesisChapter}{ABSTRACT}
\vspace*{5mm}
{\centering\bfseries ABSTRACT

\theDocTypeEN

\theTitle

\theAuthor

Karabük University\\
Institute of Graduate Programs\\
The Department of \theProgram
%The Department of \makefirstuc{\MakeLowercase{\theProgram}}%NOT workıng properly for two words department names

\theAdvisorLabelEN:\\
\theAdvisorTitle\ \theAdvisor\\
\thesisMonth\ \the\year, \thePages\ pages

}
\theAbstract

\begin{tabular}{@{}l@{\bfseries\ \ : }p{0.80\linewidth}}
\textbf{Key Words} & \theKeywords\\
\textbf{Science Code}  & \theScienceCode\\
\end{tabular}

%%%%%----------OZET---------%%%%%%%%%
\newpage%
\addxcontentsline{toc}{thesisChapter}{ÖZET}
\vspace*{8mm}
{\centering\bfseries ÖZET

\theDocTypeTR

\theTitleTR

\theAuthor

Karabük Üniversitesi\\
Lisansüstü Eğitim Enstitüsü\\
\theProgramTR\ Ana Bilim Dalı
%\makefirstuc{\MakeLowercase{\theProgramTR}}\ Ana Bilim Dalı %NOT workıng properly for two words department names

\theAdvisorLabelTR:\\
\theAdvisorTitleTR\ \theAdvisor\\
\thesisMonthTR\ \the\year, \thePages\ sayfa

}
\theOzet

\begin{tabular}{@{}l@{\bfseries\ \ : }p{0.72\linewidth}}
\textbf{Anahtar Kelimeler} & \theKeywordsTR\\
\textbf{Bilim Kodu}  & \theScienceCode\\
\end{tabular}

%%%%%%---------ACKNOWLEDGMENT----------%%%%%%%%%
\newpage%
\addxcontentsline{toc}{thesisChapter}{ACKNOWLEDGMENT}

\vspace*{9mm}
{\centering\bfseries ACKNOWLEDGMENT

}
\theAcknowledgment


%%%%------------TABLE OF CONTENTS----------%%%%%%%%%%%%%%%%%%%%%%%%
\newpage%
\addxcontentsline{toc}{thesisChapter}{CONTENTS}
\tableofcontents

%%%%------------LIST OF FIGURES----------%%%%%%%%%%%%%%%%%%%%%%%%
\newpage%
\addxcontentsline{toc}{thesisChapter}{\normalfont{LIST OF FIGURES}}
\listoffigures

%%%%------------LIST OF TABLES----------%%%%%%%%%%%%%%%%%%%%%%%%

\newpage%
\addxcontentsline{toc}{thesisChapter}{\normalfont{LIST OF TABLES}}
\listoftables

%%%%------------SYMBOLS AND ABBREVIATIONS INDEX----------%%%%%%%%%%%%%%%%%%%%%%%%

\newpage%
\addxcontentsline{toc}{thesisChapter}{\normalfont{SYMBOLS AND ABBREVIATIONS INDEX}}
\vspace*{10mm}
{\centering\bfseries SYMBOLS AND ABBREVIATIONS INDEX\\[\baselineskip]

}
\textbf{SYMBOLS}\\[-2\baselineskip]
\begin{tabbing}
\qquad\=:\=\kill
\theGetSymbols
\end{tabbing}\vspace{-1.5\baselineskip}
\textbf{ABBREVIATIONS}\\[-2\baselineskip]
\begin{tabbing}
\qquad\qquad\=:\=\kill
\theGetAbbr
\end{tabbing}

\newpage%
}

\newcommand{\makeThesisPostamble}{


%%%%%%---------REFS----------%%%%%%%%%
\newpage
\addxcontentsline{toc}{chapter}{REFERENCES}
%\printbibliography[heading={heading}]
%\printbibliography[title={\normalsize{\normalfont\textbf{REFERENCES}}}]
\vspace*{9mm}
{\centering\bfseries REFERENCES

}
%\printbibliography
%\theRefs
\printbibliography[heading=none]
%%\bibliographystyle{aichej}   % <-- The .bst file %%%% for bibtex
%%\bibliography{thesis} %%%% for bibtex

%%%%%%---------RESUME----------%%%%%%%%%
\newpage%
\addxcontentsline{toc}{thesisChapter}{CURRICULUM VITAE}

\vspace*{9mm}
{\centering\bfseries CURRICULUM VITAE

}
\theResume



}


\endinput