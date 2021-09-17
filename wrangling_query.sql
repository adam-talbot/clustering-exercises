use zillow;

select *
from properties_2017 as prop
join(
	select parcelid, max(transactiondate) as transactiondate
	from predictions_2017
	group by parcelid
	 ) as txn using(parcelid)
join predictions_2017 as pred using(parcelid, transactiondate)
left join airconditioningtype as act using (airconditioningtypeid)
left join architecturalstyletype as ast using(architecturalstyletypeid)
left join buildingclasstype as bct using(buildingclasstypeid)
left join heatingorsystemtype as hst using(heatingorsystemtypeid)
left join propertylandusetype as plt using(propertylandusetypeid)
left join storytype as st using(storytypeid)
left join typeconstructiontype as tct using(typeconstructiontypeid)
where latitude IS NOT NULL and longitude IS NOT NULL; 
;

	 