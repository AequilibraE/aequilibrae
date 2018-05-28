# coding=utf-8
class Agency:
    """
    contents of the **agency.txt** file  (from https://developers.google.com/transit/gtfs/reference/)

    Fields
    ______

    * **id** `(agency_id)` **Optional** - The agency_id field is an ID that uniquely identifies a transit agency. A transit feed may represent data from more than one agency. The agency_id is dataset unique. This field is optional for transit feeds that only contain data for a single agency.

    * **name** `(agency_name)` **Required** - The agency_name field contains the full name of the transit agency. Google Maps will display this name.

    * **url** `(agency_url)` **Required** - The agency_url field contains the URL of the transit agency. The value must be a fully qualified URL that includes http:// or https://, and any special characters in the URL must be correctly escaped. See http://www.w3.org/Addressing/URL/4_URI_Recommentations.html for a description of how to create fully qualified URL values.

    * **timezone** `(agency_timezone)` **Required** - The agency_timezone field contains the timezone where the transit agency is located. Timezone names never contain the space character but may contain an underscore. Please refer to http://en.wikipedia.org/wiki/List_of_tz_zones for a list of valid values. If multiple agencies are specified in the feed, each must have the same agency_timezone.

    * **lang** `(agency_lang)` **Optional** - The agency_lang field contains a two-letter ISO 639-1 code for the primary language used by this transit agency. The language code is case-insensitive (both en and EN are accepted). This setting defines capitalization rules and other language-specific settings for all text contained in this transit agency's feed. Please refer to http://www.loc.gov/standards/iso639-2/php/code_list.php for a list of valid values.

    * **phone** `(agency_phone)` **Optional** - The agency_phone field contains a single voice telephone number for the specified agency. This field is a string value that presents the telephone number as typical for the agency's service area. It can and should contain punctuation marks to group the digits of the number. Dialable text (for example, TriMet's "503-238-RIDE") is permitted, but the field must not contain any other descriptive text.

    * **fare_url** `(agency_url)` **Optional** - The agency_fare_url specifies the URL of a web page that allows a rider to purchase tickets or other fare instruments for that agency online. The value must be a fully qualified URL that includes http:// or https://, and any special characters in the URL must be correctly escaped. See http://www.w3.org/Addressing/URL/4_URI_Recommentations.html for a description of how to create fully qualified URL values.

    * **email** `(agency_email)` **Optional** - Contains a single valid email address actively monitored by the agency’s customer service department. This email address will be considered a direct contact point where transit riders can reach a customer service representative at the agency.
    """

    def __init__(self):
        """
        Initializes the class with members corresponding to all fields in the GTFS specification. See Stop class
        documentation
        """
        self.id = None
        self.name = None
        self.url = None
        self.timezone = None
        self.lang = None
        self.phone = None
        self.fare_url = None
        self.email = None
