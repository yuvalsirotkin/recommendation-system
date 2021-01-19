import collaborative_filtering
import content_based_filtering
import non_personalized
import Prediction_metrics


print("---non personalized--")
top_k_titles = non_personalized.get_simply_recommendation(10)
print(top_k_titles)
# print("---non personalized place--")
# top_k_titles = non_personalized.get_simply_place_recommendation('Ohio', 10)
# print(top_k_titles)
# print("---non personalized age--")
# top_k_titles = non_personalized.get_simply_age_recommendation(28, 10)
# print(top_k_titles)
# print("---collaborative filtering--")
# mat = collaborative_filtering.build_CF_prediction_matrix('cosine')
# print(mat)
# print("---collaborative filtering--")
# prediction_for_user = collaborative_filtering.get_CF_recommendation(1, 10)
# print(prediction_for_user)
# print("---contact based filtering--")
# books = content_based_filtering.get_contact_recommendation('Twilight', 10)
# print(books)
# print("----precision_k")
# precision = Prediction_metrics.precision_k(10)
# print(precision)
# print("----ARHR")
# precision = Prediction_metrics.ARHR(10)
# print(precision)
# print("----RMSE")
# precision = Prediction_metrics.RMSE()
# print(precision)
