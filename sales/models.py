from django.db import models

class Sales(models.Model):
    product = models.IntegerField(blank=True, null=True)
    date = models.DateField(blank=True, null=True)
    sales = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'sales'
