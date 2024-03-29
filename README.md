[paper]: https://aclanthology.org/2022.findings-emnlp.321
[repository]: https://github.com/the-deep/humset

HumSet is a novel and rich multilingual dataset of humanitarian response documents annotated by experts in the humanitarian response community. HumSet is curated by humanitarian analysts and covers various disasters around the globe that occurred from 2018 to 2021 in 46 humanitarian response projects. The dataset consists of approximately 17K annotated documents in three languages of English, French, and Spanish, originally taken from publicly-available resources. For each document, analysts have identified informative snippets (entries) in respect to common humanitarian frameworks, and assigned one or many classes to each entry. See the our pre-print short paper for details.

**Paper:** [Humset - Dataset of Multilingual Information Extraction and Classification for Humanitarian Crisis Response][paper]
```
@inproceedings{fekih-etal-2022-humset,
    title = "{H}um{S}et: Dataset of Multilingual Information Extraction and Classification for Humanitarian Crises Response",
    author = "Fekih, Selim  and
      Tamagnone, Nicolo{'}  and
      Minixhofer, Benjamin  and
      Shrestha, Ranjan  and
      Contla, Ximena  and
      Oglethorpe, Ewan  and
      Rekabsaz, Navid",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-emnlp.321",
    pages = "4379--4389",
}
```
<br/><br/>
* [Dataset](#dataset)
  * [Additional data](#addittional-data)

* [Request access](#request-access)
* [Contact](#contact)
* [Terms and conditions](#terms-and-conditions)
<br/><br/>

### Dataset 

Main dataset is shared in CSV format (<em>**humset_data.csv**</em>), where each row is considered as an <em>entry</em> with the following features: 

<div class="alert bg-success text-dark" cellspacing="0" style="width:100%">
  <table id="leaderboard_head_dctr" class="table table-bordered" cellspacing="0">
    <thead>
      <tr><th>entry_id</th><th>lead_id</th><th>project_id</th><th>sectors</th><th>pillars_1d</th><th>pillars_2d</th><th>subpillars_1d</th><th>subpillars_2d</th><th>lang</th><th>n_tokens</th><th>project_title</th><th>created_at</th><th>document</th><th>excerpt</th></tr>
    </thead>
  </table>
</div>

- **entry_id**: tpyeunique identification number for a given entry. (int64)
- **lead_id**: unique identification number for the document to which the corrisponding entry belongs. (int64)
- **sectors**, **pillars_1d**, **pillars_2d**, **subpillars_1d**, **subpillars_2d**: labels assigned to the corresponding entry. Since this is a multi-label dataset (each entry may have several annotations belonging to the same category), they are reported as arrays of strings. For a detailed description of these categories, see the [paper]. (list)
- **lang**: language. (str)
- **n_tokens**: number of tokens (tokenized using NLTK v3.7 library). (int64)
- **project_title**: the name of the project where the corresponding annotation was created. (str)
- **created_at**: date and time of creation of the annotation in stardard ISO 8601 format. (str)
- **document**: document URL source of the excerpt. (str)
- **excerpt**: excerpt text. (str)

**Note**: 
- **subpillars_1d** and **subpillars_2d** respective tags are reported, as strings, with the format {PILLAR}->{SUBPILLARS}, in order to underline the hierarchical structure of 1D and 2D categories. 
<br/><br/>
#### Addittional data

In addition to the main dataset, documents (<em>leads</em>) full texts are also reported (<em>**documents.tar.gz**</em>). Each text source is represented JSON-formatted file ({**lead_id**}.json) with the following structure: 
```
[
  [
    paragraph 1 - page 1,
    paragraph 2 - page 1,
    ...
    paragraph N - page 1
  ],
  [
    paragraph 1 - page 2,
    paragraph 2 - page 2,
    ...
    paragraph N - page 2
  ],
  [
    ...
  ],
  ...
]
```
Each document is a list of lists of strings, where each element is the text of a page, divided into the corresponding paragraphs. This format was used since, as indicated in the [paper], over 70% of the sources are in PDF format, thus choosing to keep the original textual subdivision. In the case of HTML web pages, the text is reported as if it belongs to a single page document.
<br/><br/>
Additionally, <em>train/validation/test</em> splitted dataset is shared. The [repository] contains the code with which it is possible to process the total dataset, but the latter contains some random components which would therefore result in a slightly different result.

## Request access
To gain access to HumSet, please contact us at [nlp@thedeep.io](mailto:nlp@thedeep.io)
<!---
fill this [form](https://docs.google.com/forms/d/e/1FAIpQLSesb1-GChU4IsUadhzyn8bJPn6usyaiICoqhqEivtkJF_zBEg/viewform)
--->
## Contact
For any technical question please contact [Selim Fekih](mailto:selim@datafriendlyspace.org), [Nicolò Tamagnone](mailto:nico@datafriendlyspace.org).

<!---
<br>
<div class="row">
    <div class="col-md-4 text-center">
        <a target="_blank" href="https://www.jku.at/en/institute-of-computational-perception/about-us/people/navid-rekab-saz/"><img src="images/navid.png" width="150" height="150"><br><strong>Navid Rekab-saz</strong><br>Johannes Kepler University Linz</a>
    </div>
    <div class="col-md-4 text-center">
        <a target="_blank" href="https://www.jku.at/en/institute-of-computational-perception/about-us/people/oleg-lesota/"><img src="images/oleg.webp" width="150" height="150"><br><strong>Oleg Lesota</strong><br>Johannes Kepler University Linz</a>
    </div>
    <div class="col-md-4 text-center">
        <a target="_blank" href="https://www.jku.at/en/institute-of-computational-perception/about-us/people/markus-schedl"><img src="images/markus.jpg" width="87" height="150"><br><strong>Markus Schedl</strong><br>Johannes Kepler University Linz</a>
    </div>
</div>
<br>
<div class="row">
    <div class="col-md-6 text-center">
        <a target="_blank" href="mailto:jon.brassey@tripdatabase.com?subject=[TripClick]"><img src="images/jon.webp" width="150" height="150"><br><strong>Jon Brassey</strong><br>Trip Database</a>
    </div>
    <div class="col-md-6 text-center">
        <a target="_blank" href="https://brown.edu/Research/AI/people/carsten.html"><img src="images/carsten.png" width="150" height="150"><br><strong>Carsten Eickhoff</strong><br>Brown University</a>
    </div>
</div>
--->

### Terms and conditions
For a detailed description about terms and conditions, refer to [DEEP Terms of Use and Privacy Notice](https://app.thedeep.io/terms-and-privacy/)

<br/><br/>
<!--- <img src="images/dfs-logo-full-color-rgb.svg" alt="DFS logo" width="90"/> --->
