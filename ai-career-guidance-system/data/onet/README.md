# O*NET source data included with Pathfinder

These selected CSV files come from the O*NET 31.0 Database, August 2026 release, supplied with the assignment. The app uses occupation descriptions, essential skills, career interest types, knowledge, education categories, job zones, and software examples.

O*NET database content is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0). Attribution: National Center for O*NET Development, O*NET 31.0 Database. License details: https://www.onetcenter.org/license_db.html. Database and file documentation: https://www.onetcenter.org/database.html.

The original source CSV headers are retained. `career_engine.py` reads the expected O*NET tables, checks required columns, ignores blank and nonnumeric rating values, and allows optional detail tables to be missing.
